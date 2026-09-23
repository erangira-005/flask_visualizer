"""
Stack and Queue implementations with full operation support.
"""


# ─────────────────────────────────────────────────────────────
#  STACK  (LIFO – Last In, First Out)
#  Think of a stack of plates: you add/remove from the TOP only
# ─────────────────────────────────────────────────────────────
class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        """Add item to the top. O(1)"""
        self._data.append(item)

    def pop(self):
        """Remove and return item from the top. O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """Return top item without removing it. O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        """Return True if stack has no items. O(1)"""
        return len(self._data) == 0

    def size(self):
        """Return number of items. O(1)"""
        return len(self._data)

    def clear(self):
        """Remove all items. O(1)"""
        self._data = []

    def to_list(self):
        """Return items as list, bottom to top. O(n)"""
        return list(self._data)

    def __repr__(self):
        return f"Stack(top → {self._data[::-1]})"


# ─────────────────────────────────────────────────────────────
#  QUEUE  (FIFO – First In, First Out)
#  Think of a queue at a shop: you join at the BACK,
#  get served from the FRONT
# ─────────────────────────────────────────────────────────────
class Queue:
    def __init__(self):
        self._data = []

    def enqueue(self, item):
        """Add item to the back. O(1)"""
        self._data.append(item)

    def dequeue(self):
        """Remove and return item from the front. O(n) with list"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.pop(0)

    def peek(self):
        """Return front item without removing it. O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[0]

    def is_empty(self):
        """Return True if queue has no items. O(1)"""
        return len(self._data) == 0

    def size(self):
        """Return number of items. O(1)"""
        return len(self._data)

    def clear(self):
        """Remove all items. O(1)"""
        self._data = []

    def to_list(self):
        """Return items as list, front to back. O(n)"""
        return list(self._data)

    def __repr__(self):
        return f"Queue(front → {self._data})"


# ─────────────────────────────────────────────────────────────
#  ALGORITHMS that USE a Stack
# ─────────────────────────────────────────────────────────────
def stack_reverse(items: list) -> list:
    """
    Reverse a list using a stack. O(n)
    Push all items on, then pop them all off — order flips.
    """
    s = Stack()
    for item in items:
        s.push(item)
    result = []
    while not s.is_empty():
        result.append(s.pop())
    return result


def stack_balanced_brackets(expression: str) -> bool:
    """
    Check if brackets are balanced using a stack. O(n)
    e.g. "({[]})" → True,  "({[})" → False
    """
    s = Stack()
    matching = {')': '(', '}': '{', ']': '['}
    for char in expression:
        if char in '({[':
            s.push(char)
        elif char in ')}]':
            if s.is_empty() or s.pop() != matching[char]:
                return False
    return s.is_empty()


def stack_sort(items: list) -> list:
    """
    Sort a list using two stacks. O(n²)
    The sorted stack always stays in ascending order (top = smallest).
    """
    input_stack = Stack()
    sorted_stack = Stack()

    for item in items:
        input_stack.push(item)

    while not input_stack.is_empty():
        temp = input_stack.pop()
        while not sorted_stack.is_empty() and sorted_stack.peek() < temp:
            input_stack.push(sorted_stack.pop())
        sorted_stack.push(temp)

    return sorted_stack.to_list()  # bottom=largest, top=smallest


def stack_evaluate_postfix(expression: str) -> float:
    """
    Evaluate a postfix (RPN) expression using a stack. O(n)
    e.g. "3 4 + 2 *" → 14  (means (3+4)*2)
    """
    s = Stack()
    for token in expression.split():
        if token in '+-*/':
            b = s.pop()
            a = s.pop()
            if token == '+':
                s.push(a + b)
            elif token == '-':
                s.push(a - b)
            elif token == '*':
                s.push(a * b)
            elif token == '/':
                s.push(a / b)
        else:
            s.push(float(token))
    return s.pop()


# ─────────────────────────────────────────────────────────────
#  ALGORITHMS that USE a Queue
# ─────────────────────────────────────────────────────────────
def queue_hot_potato(names: list, count: int) -> str:
    """
    Hot potato simulation using a queue. O(n * count)
    Pass the potato `count` times; whoever holds it is eliminated.
    Last person standing wins.
    """
    q = Queue()
    for name in names:
        q.enqueue(name)

    while q.size() > 1:
        for _ in range(count):
            q.enqueue(q.dequeue())   # pass potato
        q.dequeue()                  # eliminate front person

    return q.dequeue()


def queue_bfs(graph: dict, start: str) -> list:
    """
    Breadth-First Search using a queue. O(V + E)
    Visits nodes level by level from the start node.
    """
    visited = []
    q = Queue()
    seen = set()

    q.enqueue(start)
    seen.add(start)

    while not q.is_empty():
        node = q.dequeue()
        visited.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in seen:
                seen.add(neighbour)
                q.enqueue(neighbour)

    return visited


def queue_reverse(items: list) -> list:
    """
    Reverse a queue using a stack helper. O(n)
    """
    q = Queue()
    s = Stack()
    for item in items:
        q.enqueue(item)
    while not q.is_empty():
        s.push(q.dequeue())
    result = []
    while not s.is_empty():
        result.append(s.pop())
    return result
