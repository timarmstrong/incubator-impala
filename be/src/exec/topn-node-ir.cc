// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "exec/topn-node.h"

#include "common/compiler-util.h"
#include "util/debug-util.h"

#include "common/names.h"

using namespace impala;

void TopNNode::InsertBatchUnpartitioned(RuntimeState* state, RowBatch* batch) {
  DCHECK(!is_partitioned());
  DCHECK(!include_ties()) << "tie handling not implemented for non-partitioend top-n";
  // TODO: after inlining the comparator calls with codegen - IMPALA-4065 - we could
  // probably squeeze more performance out of this loop by ensure that as many loads
  // are hoisted out of the loop as possible (either via code changes or __restrict__)
  // annotations.
  FOREACH_ROW(batch, 0, iter) {
    bool replaced_existing_row = heap_->InsertTupleRow(this, iter.Get());
    if (replaced_existing_row) ++rows_to_reclaim_;
  }
}

bool TopNNode::Heap::InsertTupleRow(TopNNode* node, TupleRow* input_row) {
  const TupleDescriptor& tuple_desc = *node->output_tuple_desc_;
  bool replaced_existing_row = false;
  if (priority_queue_.Size() < heap_capacity()) {
    // Add all tuples until we hit capacity.
    Tuple* insert_tuple = reinterpret_cast<Tuple*>(
        node->tuple_pool_->Allocate(node->tuple_byte_size()));
    insert_tuple->MaterializeExprs<false, false>(input_row, tuple_desc,
        node->output_tuple_expr_evals_, node->tuple_pool_.get());

    priority_queue_.Push(insert_tuple);
  } else {
    // We're at capacity - compare to the first row in the priority queue to see if
    // we need to insert this row into the queue.
    DCHECK(!priority_queue_.Empty());
    Tuple* top_tuple = priority_queue_.Top();
    node->tmp_tuple_->MaterializeExprs<false, true>(input_row, tuple_desc,
        node->output_tuple_expr_evals_, nullptr);
    if (node->order_cmp_->Less(node->tmp_tuple_, top_tuple)) {
      // Pop off the old head, and replace with the new tuple. Deep copy into 'top_tuple'
      // to reuse the fixed-length memory of 'top_tuple'.
      node->tmp_tuple_->DeepCopy(top_tuple, tuple_desc, node->tuple_pool_.get());
      // Re-heapify from the top element and down.
      priority_queue_.HeapifyFromTop();
      replaced_existing_row = true;
    }
  }
  return replaced_existing_row;
}

void TopNNode::InsertBatchPartitioned(RuntimeState* state, RowBatch* batch) {
  DCHECK(is_partitioned());
  // Insert all of the rows in the batch into the per-partition heaps. The soft memory
  // limit will be checked later, in case this batch put us over the limit.
  FOREACH_ROW(batch, 0, iter) {
    tmp_tuple_->MaterializeExprs<false, true>(
        iter.Get(), *output_tuple_desc_, output_tuple_expr_evals_, nullptr);
    // TODO: IMPALA-10228: the comparator won't get inlined by codegen here.
    auto it = partition_heaps_.find(tmp_tuple_);
    Heap* new_heap = nullptr;
    Heap* heap;
    if (it == partition_heaps_.end()) {
      // Allocate the heap here, but insert in into partition_heaps_ later once we've
      // initialized the tuple that will be the key.
      new_heap =
          new Heap(*intra_partition_order_cmp_, per_partition_limit(), include_ties());
      heap = new_heap;
      COUNTER_ADD(in_mem_heap_created_counter_, 1);
    } else {
      heap = it->second.get();
    }
    heap->InsertMaterializedTuple(this, tmp_tuple_);
    if (new_heap != nullptr) {
      DCHECK_GT(new_heap->num_tuples(), 0);
      // Add the new heap with the first tuple as the key.
      partition_heaps_.emplace(new_heap->top(), unique_ptr<Heap>(new_heap));
    }
  }
}

void TopNNode::Heap::InsertMaterializedTuple(
    TopNNode* node, Tuple* materialized_tuple) {
  DCHECK(node->is_partitioned());
  const TupleDescriptor& tuple_desc = *node->output_tuple_desc_;
  Tuple* insert_tuple = nullptr;
  if (priority_queue_.Size() < heap_capacity()) {
    // Add all tuples until we hit capacity.
    insert_tuple =
        reinterpret_cast<Tuple*>(node->tuple_pool_->Allocate(node->tuple_byte_size()));
    materialized_tuple->DeepCopy(insert_tuple, tuple_desc, node->tuple_pool_.get());
    priority_queue_.Push(insert_tuple);
    return;
  }

  // We're at capacity - compare to the first row in the priority queue to see if
  // we need to insert this row into the heap.
  DCHECK(!priority_queue_.Empty());
  Tuple* top_tuple = priority_queue_.Top();
  int cmp_result =
      node->intra_partition_order_cmp_->Compare(materialized_tuple, top_tuple);
  if (!include_ties()) {
    ++num_tuples_discarded_; // One of the tuples will be discarded.
    if (cmp_result >= 0) return;
    // Pop off the old head, and replace with the new tuple. Reuse the fixed-length
    // memory of 'top_tuple' to reduce allocations.
    materialized_tuple->DeepCopy(top_tuple, tuple_desc, node->tuple_pool_.get());
    priority_queue_.HeapifyFromTop();
    return;
  }
  DCHECK(include_ties());
  // If we need to retain ties with the current head, the logic is more complex - we
  // have a logical heap in indices [0, heap_capacity()) of 'priority_queue_' plus
  // some number of tuples in 'overflowed_ties_' that are equal to priority_queue_.Top()
  // according to 'intra_partition_order_cmp_'.
  if (cmp_result == 0) {
    // This is a duplicate of the current min, we need to include it as a tie with min.
    insert_tuple =
        reinterpret_cast<Tuple*>(node->tuple_pool_->Allocate(node->tuple_byte_size()));
    materialized_tuple->DeepCopy(insert_tuple, tuple_desc, node->tuple_pool_.get());
    overflowed_ties_.push_back(insert_tuple);
  } else if (cmp_result > 0) {
    // Tuple does not belong in the heap.
    ++num_tuples_discarded_;
  } else {
    DCHECK_LT(cmp_result, 0);
    // 'materialized_tuple' needs to be added. Figure out which other tuples, if any,
    // need to be removed from the heap.
    // Pop off the head. It will compare equal with any overflowed ties at the back of
    // 'priority_queue_'
    priority_queue_.Pop();

    // Check if 'top_tuple' is tied with the new head.
    if (heap_capacity() > 1 && node->intra_partition_order_cmp_->Compare(
          top_tuple, priority_queue_.Top()) == 0) {
      // Removing all the ties would bring the heap below its capacity. We need to
      // keep all the tied tuples until we can pop them all off in one go.
      overflowed_ties_.push_back(top_tuple);
      insert_tuple =
          reinterpret_cast<Tuple*>(node->tuple_pool_->Allocate(node->tuple_byte_size()));
      materialized_tuple->DeepCopy(insert_tuple, tuple_desc, node->tuple_pool_.get());
      // Push the new tuple onto the heap, but retain the tied tuple.
      priority_queue_.Push(insert_tuple);
      return;
    } else {
      // No tuples tied with 'top_tuple' are left.
      // Reuse the fixed-length memory of 'top_tuple' to reduce allocations.
      int64_t num_rows_replaced = overflowed_ties_.size() + 1;
      num_tuples_discarded_ += num_rows_replaced;
      overflowed_ties_.clear();
      insert_tuple = top_tuple;
      materialized_tuple->DeepCopy(insert_tuple, tuple_desc, node->tuple_pool_.get());
      priority_queue_.Push(insert_tuple);
    }
  }
}

