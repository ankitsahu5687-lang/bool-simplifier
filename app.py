# Boolean Expression Simplifier
# Method used: Quine-McCluskey (Tabular method)

import itertools

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# PART 1: Read the expression typed by the user
# ------------------------------------------------------------
def make_tokens(text):
    """Turn the text into a list of single symbols."""
    text = text.replace(" ", "").upper()
    text = text.replace("|", "+").replace("&", "*").replace(".", "*")
    text = text.replace("!", "~")
    allowed = "01+*'~^()"
    tokens = []
    for ch in text:
        if ch.isalpha() or ch in allowed:
            tokens.append(ch)
        else:
            raise ValueError("Symbol '" + ch + "' is not allowed")
    if len(tokens) == 0:
        raise ValueError("Please type an expression")
    return tokens


class Parser:
    """Builds a small tree from the tokens. Order of work: OR < XOR < AND < NOT."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self):
        tree = self.parse_or()
        if self.peek() is not None:
            raise ValueError("Unexpected symbol '" + self.peek() + "'")
        return tree

    def parse_or(self):
        left = self.parse_xor()
        while self.peek() == "+":
            self.take()
            left = ("or", left, self.parse_xor())
        return left

    def parse_xor(self):
        left = self.parse_and()
        while self.peek() == "^":
            self.take()
            left = ("xor", left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while True:
            t = self.peek()
            if t == "*":                      # A*B
                self.take()
                left = ("and", left, self.parse_not())
            elif t is not None and (t.isalnum() or t in "(~"):   # AB
                left = ("and", left, self.parse_not())
            else:
                return left

    def parse_not(self):
        if self.peek() == "~":                # ~A
            self.take()
            return ("not", self.parse_not())
        node = self.parse_atom()
        while self.peek() == "'":             # A'
            self.take()
            node = ("not", node)
        return node

    def parse_atom(self):
        t = self.take()
        if t is None:
            raise ValueError("The expression ended too early")
        if t == "(":
            node = self.parse_or()
            if self.take() != ")":
                raise ValueError("Closing bracket ) is missing")
            return node
        if t in "01":
            return ("const", int(t))
        if t.isalpha():
            return ("var", t)
        raise ValueError("Unexpected symbol '" + t + "'")


def evaluate(node, values):
    """Find the value (0 or 1) of the tree for given variable values."""
    kind = node[0]
    if kind == "var":
        return values[node[1]]
    if kind == "const":
        return node[1]
    if kind == "not":
        return 1 - evaluate(node[1], values)
    a = evaluate(node[1], values)
    b = evaluate(node[2], values)
    if kind == "and":
        return a & b
    if kind == "or":
        return a | b
    return a ^ b                              # xor


def find_minterms(tree, variables):
    """Try every input combination and keep the ones where the output is 1."""
    n = len(variables)
    minterms = []
    for number in range(2 ** n):
        bits = format(number, "0" + str(n) + "b")
        values = {}
        for name, bit in zip(variables, bits):
            values[name] = int(bit)
        if evaluate(tree, values) == 1:
            minterms.append(number)
    return minterms


# ------------------------------------------------------------
# PART 2: Quine-McCluskey (Tabular) method
# ------------------------------------------------------------
def one_bit_difference(p1, p2):
    """If two patterns differ in exactly one position, return that position."""
    positions = []
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            positions.append(i)
    if len(positions) == 1:
        i = positions[0]
        if p1[i] != "-" and p2[i] != "-":
            return i
    return None


def tabular_method(minterms, n):
    """Returns the tables (columns) and the list of prime implicants."""
    # every term = (pattern like "01-1", set of minterms it covers)
    current = []
    for m in minterms:
        current.append((format(m, "0" + str(n) + "b"), {m}))

    columns = []
    primes = []

    while len(current) > 0:
        # group the terms by the number of 1s
        groups = {}
        for pattern, covered in current:
            ones = pattern.count("1")
            groups.setdefault(ones, []).append((pattern, covered))

        # combine terms of neighbouring groups
        used = set()
        new_terms = {}
        for g in sorted(groups):
            if g + 1 in groups:
                for p1, c1 in groups[g]:
                    for p2, c2 in groups[g + 1]:
                        i = one_bit_difference(p1, p2)
                        if i is not None:
                            merged = p1[:i] + "-" + p1[i + 1:]
                            new_terms[merged] = c1 | c2
                            used.add(p1)
                            used.add(p2)

        # rows of this column (for showing the table)
        rows = []
        for g in sorted(groups):
            for pattern, covered in groups[g]:
                if pattern in used:
                    status = "Combined"
                else:
                    status = "Not combined (prime implicant)"
                rows.append({
                    "Group (number of 1s)": g,
                    "Minterms": ", ".join(str(x) for x in sorted(covered)),
                    "Binary": pattern,
                    "Status": status,
                })
        columns.append(rows)

        # terms that could not be combined are prime implicants
        for pattern, covered in current:
            if pattern not in used:
                primes.append((pattern, covered))

        current = list(new_terms.items())

    return columns, primes


def count_literals(pattern):
    return len(pattern) - pattern.count("-")


def choose_cover(minterms, primes):
    """Pick the smallest set of prime implicants that covers all minterms."""
    chosen = []

    # 1) essential prime implicants: the only one covering some minterm
    for m in minterms:
        owners = [p for p in primes if m in p[1]]
        if len(owners) == 1 and owners[0] not in chosen:
            chosen.append(owners[0])

    covered = set()
    for p in chosen:
        covered = covered | p[1]
    left = [m for m in minterms if m not in covered]

    # 2) cover the remaining minterms with as few extra terms as possible
    if len(left) > 0:
        others = [p for p in primes if p not in chosen]
        for size in range(1, len(others) + 1):
            best = None
            for combo in itertools.combinations(others, size):
                got = set()
                for p in combo:
                    got = got | p[1]
                if all(m in got for m in left):
                    cost = sum(count_literals(p[0]) for p in combo)
                    if best is None or cost < best[0]:
                        best = (cost, combo)
            if best is not None:
                chosen = chosen + list(best[1])
                break
    return chosen


def pattern_to_term(pattern, variables):
    """01-1 with variables ABCD  ->  A'BD"""
    term = ""
    for bit, name in zip(pattern, variables):
        if bit == "1":
            term = term + name
        elif bit == "0":
            term = term + name + "'"
    if term == "":
        term = "1"
    return term


# ------------------------------------------------------------
# PART 3: The web page
# ------------------------------------------------------------
st.title("Boolean Expression Simplifier")
st.write("Simplification using the **Quine-McCluskey (Tabular) method**")
st.caption("Use  +  for OR,  '  for NOT,  AB  for AND,  ^  for XOR  and  ( )  for brackets.  Example:  A'B + AB' + AB")

text = st.text_input("Enter Boolean expression", "A'B + AB' + AB")

if st.button("Simplify"):
    try:
        tree = Parser(make_tokens(text)).parse()
        variables = sorted(set(ch for ch in make_tokens(text) if ch.isalpha()))
        n = len(variables)

        if n == 0:
            st.success("Simplified expression:  F = " + str(evaluate(tree, {})))
        elif n > 6:
            st.error("Please use at most 6 different variables.")
        else:
            minterms = find_minterms(tree, variables)

            if len(minterms) == 0:
                st.success("Simplified expression:  F = 0")
                st.write("The expression is never 1, so there is nothing to simplify.")
            else:
                columns, primes = tabular_method(minterms, n)
                chosen = choose_cover(minterms, primes)
                terms = [pattern_to_term(p[0], variables) for p in chosen]
                result = " + ".join(sorted(terms))

                st.success("Simplified expression:  F = " + result)

                # quick check that the answer is correct
                correct = True
                for number in range(2 ** n):
                    bits = format(number, "0" + str(n) + "b")
                    inside = False
                    for p in chosen:
                        if all(a == "-" or a == b for a, b in zip(p[0], bits)):
                            inside = True
                    if inside != (number in minterms):
                        correct = False
                if correct:
                    st.caption("Checked: the answer gives the same output as the original expression.")

                # show the tabular method
                st.subheader("Tabular method")
                st.write("Minterms (rows where F = 1): " + ", ".join(str(m) for m in minterms))
                for number, rows in enumerate(columns, start=1):
                    st.write("**Column " + str(number) + "**")
                    st.dataframe(pd.DataFrame(rows), hide_index=True)

    except Exception as error:
        st.error("Invalid expression: " + str(error))
