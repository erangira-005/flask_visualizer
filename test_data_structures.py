"""
Full test suite for Stack, Queue, and their algorithms.
Run with:  pytest test_data_structures.py -v
"""

import pytest
from data_structures import (
    Stack, Queue,
    stack_reverse, stack_balanced_brackets,
    stack_sort, stack_evaluate_postfix,
    queue_hot_potato, queue_bfs, queue_reverse,
)


# ═══════════════════════════════════════════════════════════════
#  STACK TESTS
# ═══════════════════════════════════════════════════════════════
class TestStack:

    # ── Basic operations ────────────────────────────────────────
    def test_new_stack_is_empty(self):
        s = Stack()
        assert s.is_empty() is True

    def test_new_stack_size_is_zero(self):
        s = Stack()
        assert s.size() == 0

    def test_push_increases_size(self):
        s = Stack()
        s.push(1)
        assert s.size() == 1
        s.push(2)
        assert s.size() == 2

    def test_push_makes_stack_not_empty(self):
        s = Stack()
        s.push("hello")
        assert s.is_empty() is False

    def test_pop_returns_last_pushed(self):
        s = Stack()
        s.push(10)
        s.push(20)
        s.push(30)
        assert s.pop() == 30

    def test_pop_lifo_order(self):
        s = Stack()
        for i in [1, 2, 3, 4, 5]:
            s.push(i)
        result = []
        while not s.is_empty():
            result.append(s.pop())
        assert result == [5, 4, 3, 2, 1]

    def test_pop_decreases_size(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.pop()
        assert s.size() == 1

    def test_pop_empty_raises_error(self):
        s = Stack()
        with pytest.raises(IndexError):
            s.pop()

    def test_peek_returns_top_without_removing(self):
        s = Stack()
        s.push(42)
        s.push(99)
        assert s.peek() == 99
        assert s.size() == 2   # size unchanged

    def test_peek_empty_raises_error(self):
        s = Stack()
        with pytest.raises(IndexError):
            s.peek()

    def test_clear_empties_stack(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        s.clear()
        assert s.is_empty() is True
        assert s.size() == 0

    def test_to_list_order(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        assert s.to_list() == [1, 2, 3]   # bottom to top

    def test_push_various_types(self):
        s = Stack()
        s.push(1)
        s.push("hello")
        s.push([1, 2, 3])
        s.push(None)
        assert s.size() == 4
        assert s.pop() is None
        assert s.pop() == [1, 2, 3]

    def test_single_item_push_pop(self):
        s = Stack()
        s.push("only")
        assert s.pop() == "only"
        assert s.is_empty()

    def test_repr(self):
        s = Stack()
        s.push(1)
        assert "Stack" in repr(s)


# ── Stack algorithm tests ────────────────────────────────────
class TestStackAlgorithms:

    def test_stack_reverse_basic(self):
        assert stack_reverse([1, 2, 3, 4, 5]) == [5, 4, 3, 2, 1]

    def test_stack_reverse_single(self):
        assert stack_reverse([42]) == [42]

    def test_stack_reverse_empty(self):
        assert stack_reverse([]) == []

    def test_stack_reverse_strings(self):
        assert stack_reverse(["a", "b", "c"]) == ["c", "b", "a"]

    def test_balanced_brackets_valid(self):
        assert stack_balanced_brackets("()") is True
        assert stack_balanced_brackets("()[]{}") is True
        assert stack_balanced_brackets("({[]})") is True
        assert stack_balanced_brackets("{[()]}") is True

    def test_balanced_brackets_invalid(self):
        assert stack_balanced_brackets("(]") is False
        assert stack_balanced_brackets("([)]") is False
        assert stack_balanced_brackets("{[}") is False
        assert stack_balanced_brackets("(((") is False

    def test_balanced_brackets_empty_string(self):
        assert stack_balanced_brackets("") is True

    def test_balanced_brackets_no_brackets(self):
        assert stack_balanced_brackets("hello world") is True

    def test_stack_sort_ascending(self):
        result = stack_sort([3, 1, 4, 1, 5, 9, 2, 6])
        assert result == sorted([3, 1, 4, 1, 5, 9, 2, 6], reverse=True)

    def test_stack_sort_already_sorted(self):
        result = stack_sort([1, 2, 3, 4, 5])
        assert result == [5, 4, 3, 2, 1]

    def test_stack_sort_reverse_sorted(self):
        result = stack_sort([5, 4, 3, 2, 1])
        assert result == [5, 4, 3, 2, 1]

    def test_stack_sort_single(self):
        assert stack_sort([7]) == [7]

    def test_stack_sort_empty(self):
        assert stack_sort([]) == []

    def test_postfix_addition(self):
        assert stack_evaluate_postfix("3 4 +") == 7.0

    def test_postfix_subtraction(self):
        assert stack_evaluate_postfix("10 3 -") == 7.0

    def test_postfix_multiplication(self):
        assert stack_evaluate_postfix("3 4 + 2 *") == 14.0

    def test_postfix_division(self):
        assert stack_evaluate_postfix("8 2 /") == 4.0

    def test_postfix_complex(self):
        # (5 + 3) * (2 - 1) = 8
        assert stack_evaluate_postfix("5 3 + 2 1 - *") == 8.0


# ═══════════════════════════════════════════════════════════════
#  QUEUE TESTS
# ═══════════════════════════════════════════════════════════════
class TestQueue:

    # ── Basic operations ────────────────────────────────────────
    def test_new_queue_is_empty(self):
        q = Queue()
        assert q.is_empty() is True

    def test_new_queue_size_is_zero(self):
        q = Queue()
        assert q.size() == 0

    def test_enqueue_increases_size(self):
        q = Queue()
        q.enqueue(1)
        assert q.size() == 1
        q.enqueue(2)
        assert q.size() == 2

    def test_enqueue_makes_queue_not_empty(self):
        q = Queue()
        q.enqueue("hello")
        assert q.is_empty() is False

    def test_dequeue_returns_first_enqueued(self):
        q = Queue()
        q.enqueue(10)
        q.enqueue(20)
        q.enqueue(30)
        assert q.dequeue() == 10

    def test_dequeue_fifo_order(self):
        q = Queue()
        for i in [1, 2, 3, 4, 5]:
            q.enqueue(i)
        result = []
        while not q.is_empty():
            result.append(q.dequeue())
        assert result == [1, 2, 3, 4, 5]

    def test_dequeue_decreases_size(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.dequeue()
        assert q.size() == 1

    def test_dequeue_empty_raises_error(self):
        q = Queue()
        with pytest.raises(IndexError):
            q.dequeue()

    def test_peek_returns_front_without_removing(self):
        q = Queue()
        q.enqueue(42)
        q.enqueue(99)
        assert q.peek() == 42
        assert q.size() == 2   # size unchanged

    def test_peek_empty_raises_error(self):
        q = Queue()
        with pytest.raises(IndexError):
            q.peek()

    def test_clear_empties_queue(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.clear()
        assert q.is_empty() is True
        assert q.size() == 0

    def test_to_list_order(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        assert q.to_list() == [1, 2, 3]   # front to back

    def test_enqueue_various_types(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue("hello")
        q.enqueue({"key": "val"})
        assert q.size() == 3
        assert q.dequeue() == 1

    def test_single_item_enqueue_dequeue(self):
        q = Queue()
        q.enqueue("only")
        assert q.dequeue() == "only"
        assert q.is_empty()

    def test_repr(self):
        q = Queue()
        q.enqueue(1)
        assert "Queue" in repr(q)


# ── Queue algorithm tests ────────────────────────────────────
class TestQueueAlgorithms:

    def test_hot_potato_returns_string(self):
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        winner = queue_hot_potato(names, 3)
        assert isinstance(winner, str)
        assert winner in names

    def test_hot_potato_single_player(self):
        assert queue_hot_potato(["Alice"], 5) == "Alice"

    def test_hot_potato_deterministic(self):
        # Same input always gives same winner
        names = ["A", "B", "C", "D", "E"]
        assert queue_hot_potato(names, 7) == queue_hot_potato(names, 7)

    def test_bfs_simple_graph(self):
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D", "E"],
            "D": [],
            "E": [],
        }
        result = queue_bfs(graph, "A")
        # A must come first
        assert result[0] == "A"
        # All nodes visited
        assert set(result) == {"A", "B", "C", "D", "E"}
        # B and C before D and E (level order)
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("E")

    def test_bfs_single_node(self):
        graph = {"A": []}
        assert queue_bfs(graph, "A") == ["A"]

    def test_bfs_linear_graph(self):
        graph = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
        assert queue_bfs(graph, "A") == ["A", "B", "C", "D"]

    def test_queue_reverse_basic(self):
        assert queue_reverse([1, 2, 3, 4, 5]) == [5, 4, 3, 2, 1]

    def test_queue_reverse_single(self):
        assert queue_reverse([42]) == [42]

    def test_queue_reverse_empty(self):
        assert queue_reverse([]) == []

    def test_queue_reverse_strings(self):
        assert queue_reverse(["x", "y", "z"]) == ["z", "y", "x"]
