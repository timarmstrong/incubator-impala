package org.apache.impala.planner;

import java.util.ArrayList;
import java.util.List;

import org.apache.impala.analysis.Analyzer;
import org.apache.impala.analysis.CompoundPredicate;
import org.apache.impala.analysis.CompoundPredicate.Operator;
import org.apache.impala.analysis.Expr;
import org.apache.impala.analysis.ExprSubstitutionMap;
import org.apache.impala.analysis.SlotDescriptor;
import org.apache.impala.analysis.SlotId;
import org.apache.impala.analysis.SlotRef;
import org.apache.impala.analysis.TupleDescriptor;
import org.apache.impala.analysis.TupleId;
import org.apache.impala.analysis.TupleIsNullPredicate;
import org.apache.impala.common.ImpalaException;
import org.apache.impala.common.TreeNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.Lists;

/**
 * Implementation of slot projection optimization that is applied to a distributed plan.
 */
public class SlotProjectionPass {
  private static final Logger LOG = LoggerFactory.getLogger(SlotProjectionPass.class);

  /**
   * Apply the projection trimming optimisation to the distributed plan rooted at 'rootFragment'.
   * This will modify the plan in-place and substitute tuples and expressions to use the projected
   * tuples. Returns the updated 'resultExprs' with any required substitutions.
   */
  public static List<Expr> applyProjection(PlannerContext ctx, PlanFragment rootFragment,
      List<Expr> resultExprs)
      throws ImpalaException {
    ExprSubstitutionMap rootNodeSmap;
    applyProjectionImpl(ctx, rootFragment.getPlanRoot(), resultExprs);
    rootNodeSmap = rootFragment.getPlanRoot().getOutputSmap();
    resultExprs = Expr.substituteList(
        resultExprs, rootNodeSmap, ctx.getRootAnalyzer(), true);
    rootFragment.setOutputExprs(resultExprs);
    return resultExprs;
  }

  private static class StackEntry {
    public StackEntry(PlanNode node) {
      this.node = node;
    }
    // The node itself.
    public final PlanNode node;

    // The number of expressions that were in the expression stack before visiting
    // this node. -1 if the node has not yet been visited (and therefore does not have
    // expressions in the expr stack).
    public int savedExprStackSize = -1;

    public boolean visited() {
      return savedExprStackSize != -1;
    }
  }

  private static void applyProjectionImpl(
      PlannerContext ctx, PlanNode root, List<Expr> resultExprs) throws ImpalaException {
    ArrayList<StackEntry> workStack = new ArrayList<>();
    ArrayList<Expr> exprStack = new ArrayList<>(resultExprs);

    // Do a DFS on the tree, visiting each node both before and after its descendants.
    // We use two stacks - one for the nodes being visited and one for the expressions
    // in all of the descendant nodes.
    // TODO: clarify which exprs we actually need to track so that we can reduce the
    // amount of processing required.
    workStack.add(new StackEntry(root));
    while (!workStack.isEmpty()) {
      StackEntry e = workStack.remove(workStack.size() - 1);
      if (e.visited()) {
        Preconditions.checkState(exprStack.size() >= e.savedExprStackSize);

        applyProjectionToNode(ctx, e.node, exprStack);

        // Remove all of this node's expressions from the stack.
        while (exprStack.size() > e.savedExprStackSize) {
          exprStack.remove(exprStack.size() - 1);
        }
      } else {
       // First time visiting this node. Add exprs and process descendants in DFS order.
        e.savedExprStackSize = exprStack.size();
        e.node.collectExprsWithSlotRefs(exprStack);

        // Visit all descendents in DFS order, then this node again.
        workStack.add(e);
        for (PlanNode child: e.node.getChildren()) {
          workStack.add(new StackEntry(child));
        }
      }
    }
  }

  /**
   * Tries to apply projection to 'node'.
   * @param node is the node to attempt to apply projection to.
   * @param parentExprs are all of the expressions from the parent and other
   *        ancestors of this node, used to compute required slots.
   */
  private static void applyProjectionToNode(PlannerContext ctx, PlanNode node,
      ArrayList<Expr> parentExprs) throws ImpalaException {
    Analyzer analyzer = ctx.getRootAnalyzer();

    // TODO: needs explanation. I think we only need to do the TupleIsNullPredicate
    // substitution at this point, if at all.
    node.substituteExprs(node.getCombinedChildSmap(), analyzer);

    ProjectionInfo projection = null;
    if (node instanceof ExchangeNode) {
      ExchangeNode exchNode = (ExchangeNode) node;
      List<Expr> substParentExprs = Expr.substituteList(
          parentExprs, exchNode.getChild(0).getOutputSmap(), analyzer, true);
      projection = computeProjection(
          analyzer, exchNode.getChild(0).getTupleIds(), substParentExprs);
      if (projection != null) {
        PlanFragment inputFragment = node.getChild(0).getFragment();
        // TODO: implement the materialisation in the exchange
        UnionNode unionNode = UnionNode.createProjection(ctx.getNextNodeId(),
            projection.tupleDesc.getId(), projection.smap.getRhs(), false);
        unionNode.addChild(inputFragment.getPlanRoot(), projection.smap.getLhs());
        unionNode.setOutputSmap(ExprSubstitutionMap.compose(
            inputFragment.getPlanRoot().getOutputSmap(), projection.smap, analyzer));
        unionNode.init(analyzer);
        inputFragment.setPlanRoot(unionNode);
        node.setChild(0, unionNode);
      }
    }

    // Compute row composition, apply child smaps and re-compute stats.
    ExprSubstitutionMap smap = node.getCombinedChildSmap();
    node.substituteExprs(smap, analyzer);
    node.setOutputSmap(smap);
    node.computeTupleIds();
    node.validateExprs(); // for debugging
    node.computeStats(analyzer);
  }

  private static class ProjectionInfo {
    public ProjectionInfo(TupleDescriptor tupleDesc, ExprSubstitutionMap smap) {
      this.tupleDesc = tupleDesc;
      this.smap = smap;
    }

    // New tuple with only projected slots.
    public final TupleDescriptor tupleDesc;

    // Map from the old pre-projection expressions to expressions evaluated over the new
    // slots.
    public final ExprSubstitutionMap smap;
  }

  /**
   * Compute a new tuple layout with only the slots from 'tids' required to evaluate
   * 'exprs'.
   */
  private static ProjectionInfo computeProjection(
      Analyzer analyzer, List<TupleId> tids, List<Expr> exprs) {
    ComputedSlots slots = computeRequiredSlots(analyzer, tids, exprs);

    // Only apply projection if it will reduce the number of slots or flatten multiple
    // tuples into a single tuple.
    if (slots.usedSlotDescs.size() == slots.numMaterializedSlots &&
        tids.size() < 2) {
      return null;
    }
    TupleDescriptor projectedTuple =
        analyzer.getDescTbl().createTupleDescriptor("projection");
    ExprSubstitutionMap projectionSmap = new ExprSubstitutionMap();
    for (SlotDescriptor slotDesc: slots.usedSlotDescs) {
      SlotDescriptor projectedSlotDesc = analyzer.copySlotDescriptor(slotDesc, projectedTuple);
      projectedSlotDesc.setIsMaterialized(true);
      projectedSlotDesc.setSourceExpr(new SlotRef(slotDesc));
      projectionSmap.put(new SlotRef(slotDesc), new SlotRef(projectedSlotDesc));
    }

    // Materialize TupleIsNullPredicates that refer to tuples from this subtree.
    List<TupleIsNullPredicate> tupleIsNullPreds = Lists.newArrayList();
    TreeNode.collect(exprs, Predicates.instanceOf(TupleIsNullPredicate.class), tupleIsNullPreds);
    Expr.removeDuplicates(tupleIsNullPreds);
    for (TupleIsNullPredicate tupleIsNullPred: tupleIsNullPreds) {
      LOG.info("Check " + tupleIsNullPred.toSql());
      ArrayList<TupleId> boundTupleIds = new ArrayList<>();
      ArrayList<TupleId> unboundTupleIds = new ArrayList<>();
      for (TupleId referencedTid : tupleIsNullPred.getTupleIds()) {
        if (tids.contains(referencedTid)) {
          boundTupleIds.add(referencedTid);
        } else {
          unboundTupleIds.add(referencedTid);
        }
      }
      if (boundTupleIds.isEmpty()) continue;
      // We need to replace TupleIsNullPredicate(...) with
      // TODO: update comment to reflect below logic
      // SlotRef(...) && TupleIsNullPredicate(unboundTupleIds, newTupleId), where
      // SlotRef contains the result of evaluating TupleIsNullPredicate(boundTupleIds).
      SlotDescriptor slotDesc = analyzer.addSlotDescriptor(projectedTuple);
      TupleIsNullPredicate exprToEval = new TupleIsNullPredicate(boundTupleIds);
      exprToEval.analyzeNoThrow(analyzer);
      slotDesc.initFromExpr(exprToEval);
      slotDesc.setIsMaterialized(true);

      // unboundTupleIds.add(projectedTuple.getId());
      Expr replacementExpr = new CompoundPredicate(Operator.OR, new SlotRef(slotDesc),
          new TupleIsNullPredicate(projectedTuple.getId()));
      if (!unboundTupleIds.isEmpty()) {
        replacementExpr = new CompoundPredicate(Operator.AND,
          replacementExpr, new TupleIsNullPredicate(unboundTupleIds));
      }
      replacementExpr.analyzeNoThrow(analyzer);
      projectionSmap.put(tupleIsNullPred.clone(), replacementExpr);
    }

    projectedTuple.computeMemLayout();
    return new ProjectionInfo(projectedTuple, projectionSmap);
  }

  /**
   * Return value for computeRequiredSlots().
   */
  private static class ComputedSlots {
    public ComputedSlots(List<SlotDescriptor> usedSlotDescs, long numMaterializedSlots) {
      Preconditions.checkState(usedSlotDescs.size() <= numMaterializedSlots);
      this.usedSlotDescs = usedSlotDescs;
      this.numMaterializedSlots = numMaterializedSlots;
    }

    final List<SlotDescriptor> usedSlotDescs;
    final long numMaterializedSlots;
  }

  /**
   * Helper for computeProjection() that computes the slots from 'tids' that are
   * required to compute 'exprs'.
   */
  private static ComputedSlots computeRequiredSlots(Analyzer analyzer, List<TupleId> tids,
      List<Expr> exprs) {
    // Slot ids required to evaluate exprs.
    List<SlotId> projectedSids = Lists.newArrayList();
    Expr.getIds(exprs, null, projectedSids);

    long numMaterializedSlots = 0;
    List<SlotDescriptor> usedSlotDescs = Lists.newArrayList();
    for (TupleId tid: tids) {
      TupleDescriptor tupleDesc = analyzer.getTupleDesc(tid);
      for (SlotDescriptor slotDesc: tupleDesc.getSlots()) {
        if (projectedSids.contains(slotDesc.getId())) {
          Preconditions.checkState(slotDesc.isMaterialized(), slotDesc);
          usedSlotDescs.add(slotDesc);
        }
        if (slotDesc.isMaterialized()) ++numMaterializedSlots;
      }
    }
    return new ComputedSlots(usedSlotDescs, numMaterializedSlots);
  }

}
