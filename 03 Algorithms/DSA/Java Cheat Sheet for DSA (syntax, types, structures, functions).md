---
base: "[[DSA.base]]"
Order: 0
Type: Concept
Source: Lesson Summary
Topic: []
Status: To Review
Category: Fundamentals
---
A quick reference for solving DSA problems in Java: core syntax, data types, the collection classes you'll reach for, and the methods used most often.

## 1. Primitive types, ranges & wrappers

| Type | Size | Range / note | Wrapper |
| --- | --- | --- | --- |
| `int` | 32-bit | ±2.1×10^9 (`Integer.MAX_VALUE`=2147483647) | `Integer` |
| `long` | 64-bit | ±9.2×10^18 (suffix `L`) | `Long` |
| `double` | 64-bit | floating point | `Double` |
| `char` | 16-bit | a character; `'a'-'a'=0`, arithmetic allowed | `Character` |
| `boolean` | — | true/false | `Boolean` |
| `byte`/`short` | 8/16-bit | rarely needed | — |

```java
int maxInt = Integer.MAX_VALUE, minInt = Integer.MIN_VALUE;
long big = 1_000_000_007L;        // use long to avoid overflow when summing
int digit = '7' - '0';            // 7 ; char arithmetic
int idx = c - 'a';                // map letter to 0..25
```

## 2. Syntax essentials

```java
// loops
for (int i = 0; i < n; i++) { }
for (int x : arr) { }                     // enhanced for
while (cond) { }

// conditional / ternary
int m = (a > b) ? a : b;

// method
static int add(int a, int b) { return a + b; }

// lambda + comparator
Comparator<int[]> bySecond = (x, y) -> Integer.compare(x[1], y[1]);
Runnable r = () -> System.out.println("hi");
```

## 3. Arrays

```java
int[] a = new int[n];                 // zero-filled
int[] b = {5, 3, 1};
int[][] grid = new int[rows][cols];

Arrays.sort(a);                       // ascending, O(n log n)
Arrays.sort(b, 0, 2);                 // sort sub-range [0,2)
Arrays.fill(a, -1);
int[] copy = Arrays.copyOf(a, a.length);
int[] slice = Arrays.copyOfRange(a, l, r); // [l, r)
Arrays.toString(a);                   // printable
// descending: sort boxed then reverse, or sort & swap. Primitives can't use Comparator.
Integer[] boxed = {5,3,1};
Arrays.sort(boxed, Collections.reverseOrder());
```

## 4. Strings

```java
String s = "hello";
s.length(); s.charAt(0); s.substring(1, 3); // "el", [1,3)
s.indexOf('l'); s.contains("ell");
s.equals("hello"); s.compareTo("world");
char[] cs = s.toCharArray();
String[] parts = "a,b,c".split(",");
String joined = String.join("-", parts);
Integer.parseInt("42"); String.valueOf(42);

// Strings are immutable — build with StringBuilder for O(n)
StringBuilder sb = new StringBuilder();
sb.append('x').append(42);
sb.reverse(); sb.deleteCharAt(sb.length()-1);
String out = sb.toString();
```

## 5. List (dynamic array)

```java
List<Integer> list = new ArrayList<>();
list.add(1); list.add(0, 9);     // append / insert at index
list.get(0); list.set(0, 5);
list.remove(0);                  // by index
list.remove(Integer.valueOf(5)); // by value
list.size(); list.contains(5); list.isEmpty();
Collections.sort(list);
Collections.sort(list, (x, y) -> y - x);   // descending
Collections.reverse(list);
```

## 6. Set (membership / dedup)

```java
Set<Integer> set = new HashSet<>();   // O(1) avg, unordered
set.add(3); set.contains(3); set.remove(3);
if (!set.add(x)) { /* x already present (duplicate) */ }

TreeSet<Integer> ts = new TreeSet<>(); // sorted, O(log n)
ts.first(); ts.last();
ts.floor(4);    // largest <= 4
ts.ceiling(4);  // smallest >= 4
ts.higher(4); ts.lower(4);            // strict
```

## 7. Map (key → value)

```java
Map<String, Integer> map = new HashMap<>();
map.put("a", 1);
map.get("a");
map.getOrDefault("b", 0);
map.containsKey("a"); map.containsValue(1);
map.merge("a", 1, Integer::sum);            // frequency count idiom
map.computeIfAbsent("k", x -> new ArrayList<>()).add(7); // multimap idiom
map.keySet(); map.values(); map.entrySet();
for (Map.Entry<String,Integer> e : map.entrySet()) { e.getKey(); e.getValue(); }

TreeMap<Integer,String> tm = new TreeMap<>(); // sorted keys, O(log n)
tm.firstKey(); tm.lastKey(); tm.floorKey(4); tm.ceilingKey(4);
```

## 8. Stack & Queue & Deque (use ArrayDeque)

```java
Deque<Integer> stack = new ArrayDeque<>();   // LIFO
stack.push(1); stack.peek(); stack.pop();

Queue<Integer> queue = new ArrayDeque<>();    // FIFO
queue.offer(1); queue.peek(); queue.poll();

Deque<Integer> dq = new ArrayDeque<>();        // both ends
dq.offerFirst(0); dq.offerLast(9);
dq.pollFirst(); dq.pollLast(); dq.peekFirst(); dq.peekLast();
```

## 9. PriorityQueue (heap)

```java
PriorityQueue<Integer> min = new PriorityQueue<>();                 // min-heap
PriorityQueue<Integer> max = new PriorityQueue<>(Collections.reverseOrder());
PriorityQueue<int[]> pq = new PriorityQueue<>((x, y) -> x[1] - y[1]); // by field
min.offer(5); min.peek(); min.poll(); min.size();
```

## 10. Math & utility functions

```java
Math.max(a, b); Math.min(a, b); Math.abs(x);
Math.pow(2, 10); Math.sqrt(n); Math.ceil(x); Math.floor(x);
Math.floorDiv(a, b); Math.floorMod(a, b);   // correct for negatives
Integer.parseInt(s); Integer.toBinaryString(n);
Integer.bitCount(n);                         // popcount
Long.parseLong(s); Character.isDigit(c); Character.isLetter(c);
Character.toLowerCase(c); Character.getNumericValue(c);
```

## 11. Common idioms

```java
// frequency map
Map<Character,Integer> freq = new HashMap<>();
for (char c : s.toCharArray()) freq.merge(c, 1, Integer::sum);

// 2D visited
boolean[][] seen = new boolean[m][n];
int[][] dirs = {{1,0},{-1,0},{0,1},{0,-1}};   // grid moves

// sort 2D array by first then second
Arrays.sort(intervals, (x, y) -> x[0] != y[0] ? x[0]-y[0] : x[1]-y[1]);

// avoid overflow comparing: use Integer.compare(a,b) not a-b for large values
```

## Cheat: which structure when

- Fast membership / dedup → `HashSet`; ordered → `TreeSet`.
- Count / lookup by key → `HashMap`; ordered keys / floor-ceiling → `TreeMap`.
- LIFO → `ArrayDeque` as stack; FIFO / BFS → `ArrayDeque` as queue.
- Repeatedly get min/max → `PriorityQueue`.
- Index access + resize → `ArrayList`. Frequent middle insert/delete with a node handle → linked list.