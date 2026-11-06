# Boolean Expression Simplifier
# Method used: Quine-McCluskey (Tabular method)
# Output: minimal SOP, minimal POS, canonical forms, truth table and dashboard

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
                    "Terms": ", ".join(str(x) for x in sorted(covered)),
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


# ------------------------------------------------------------
# PART 3: Writing the answer as SOP and POS
# ------------------------------------------------------------
def pattern_to_term(pattern, variables):
    """SOP product term:  01-1 with variables ABCD  ->  A'BD"""
    term = ""
    for bit, name in zip(pattern, variables):
        if bit == "1":
            term = term + name
        elif bit == "0":
            term = term + name + "'"
    if term == "":
        term = "1"
    return term


def pattern_to_sum(pattern, variables):
    """POS sum term (for a maxterm pattern):  01-1 with ABCD  ->  (A + B' + D')
    In a sum term, 0 means the plain variable and 1 means the complemented one."""
    literals = []
    for bit, name in zip(pattern, variables):
        if bit == "0":
            literals.append(name)
        elif bit == "1":
            literals.append(name + "'")
    if len(literals) == 0:
        return "0"
    if len(literals) == 1:
        return literals[0]
    return "(" + " + ".join(literals) + ")"


def build_sop(minterms, n, variables):
    """Minimal Sum of Products. Returns (text, columns, chosen)."""
    if len(minterms) == 0:
        return "0", [], []
    columns, primes = tabular_method(minterms, n)
    chosen = choose_cover(minterms, primes)
    terms = [pattern_to_term(p[0], variables) for p in chosen]
    return " + ".join(sorted(terms)), columns, chosen


def build_pos(maxterms, n, variables):
    """Minimal Product of Sums (simplify the zeros of F). Returns (text, columns, chosen)."""
    if len(maxterms) == 0:
        return "1", [], []
    columns, primes = tabular_method(maxterms, n)
    chosen = choose_cover(maxterms, primes)
    terms = [pattern_to_sum(p[0], variables) for p in chosen]
    if "0" in terms:
        return "0", columns, chosen
    return "".join(sorted(terms)), columns, chosen


def covered_numbers(chosen, n):
    """All input numbers that are covered by the chosen patterns."""
    result = set()
    for number in range(2 ** n):
        bits = format(number, "0" + str(n) + "b")
        for p in chosen:
            if all(a == "-" or a == b for a, b in zip(p[0], bits)):
                result.add(number)
                break
    return result


def total_literals(chosen):
    return sum(count_literals(p[0]) for p in chosen)


def canonical_sop(minterms, n, variables):
    if len(minterms) == 0:
        return "0"
    return " + ".join(pattern_to_term(format(m, "0" + str(n) + "b"), variables) for m in minterms)


def canonical_pos(maxterms, n, variables):
    if len(maxterms) == 0:
        return "1"
    return "".join(pattern_to_sum(format(m, "0" + str(n) + "b"), variables) for m in maxterms)


def number_list(numbers):
    return ",".join(str(x) for x in numbers)


# ------------------------------------------------------------
# PART 4: The web page (dashboard)
# ------------------------------------------------------------
st.set_page_config(page_title="Boolean Expression Simplifier", layout="wide")

st.title("Boolean Expression Simplifier")
st.write("Simplification using the **Quine-McCluskey (Tabular) method**  -  SOP and POS dashboard")
st.caption("Use  +  for OR,  '  for NOT,  AB  for AND,  ^  for XOR  and  ( )  for brackets.  Example:  A'B + AB' + AB")

text = st.text_input("Enter Boolean expression", "A'B + AB' + AB")

if st.button("Simplify"):
    try:
        tokens = make_tokens(text)
        tree = Parser(tokens).parse()
        variables = sorted(set(ch for ch in tokens if ch.isalpha()))
        n = len(variables)

        if n == 0:
            st.success("Simplified expression:  F = " + str(evaluate(tree, {})))
        elif n > 6:
            st.error("Please use at most 6 different variables.")
        else:
            minterms = find_minterms(tree, variables)
            maxterms = [i for i in range(2 ** n) if i not in minterms]

            sop_text, sop_columns, sop_chosen = build_sop(minterms, n, variables)
            pos_text, pos_columns, pos_chosen = build_pos(maxterms, n, variables)

            # quick checks that both answers are correct
            sop_ok = (len(minterms) == 0) or (covered_numbers(sop_chosen, n) == set(minterms))
            pos_ok = (len(maxterms) == 0) or (covered_numbers(pos_chosen, n) == set(maxterms))

            # ---------- Dashboard: result cards ----------
            st.subheader("Dashboard")
            col1, col2 = st.columns(2)
            with col1:
                st.success("Minimal SOP:   F = " + sop_text)
            with col2:
                st.success("Minimal POS:   F = " + pos_text)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Variables", n)
            m2.metric("Minterms (F = 1)", len(minterms))
            m3.metric("Maxterms (F = 0)", len(maxterms))
            m4.metric("SOP literals", total_literals(sop_chosen))
            m5.metric("POS literals", total_literals(pos_chosen))

            if sop_ok and pos_ok:
                st.caption("Checked: both SOP and POS give the same output as the original expression.")
            else:
                st.warning("Check failed: please verify the result manually.")

            # ---------- Dashboard: SOP / POS details + truth table ----------
            left, right = st.columns([3, 2])

            with left:
                st.markdown("#### SOP and POS forms")
                forms = pd.DataFrame([
                    {"Form": "Canonical SOP",
                     "Notation": "Σm(" + number_list(minterms) + ")",
                     "Expression": canonical_sop(minterms, n, variables)},
                    {"Form": "Minimal SOP",
                     "Notation": "Σm(" + number_list(minterms) + ")",
                     "Expression": sop_text},
                    {"Form": "Canonical POS",
                     "Notation": "ΠM(" + number_list(maxterms) + ")",
                     "Expression": canonical_pos(maxterms, n, variables)},
                    {"Form": "Minimal POS",
                     "Notation": "ΠM(" + number_list(maxterms) + ")",
                     "Expression": pos_text},
                ])
                st.dataframe(forms, hide_index=True, width="stretch")

            with right:
                st.markdown("#### Truth table")
                rows = []
                for number in range(2 ** n):
                    bits = format(number, "0" + str(n) + "b")
                    row = {"No.": number}
                    for name, bit in zip(variables, bits):
                        row[name] = int(bit)
                    row["F"] = 1 if number in minterms else 0
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            # ---------- Tabular method tables ----------
            st.subheader("Tabular method")
            tab_sop, tab_pos = st.tabs(["SOP (using minterms)", "POS (using maxterms)"])

            with tab_sop:
                if len(minterms) == 0:
                    st.write("The expression is never 1, so F = 0.")
                else:
                    st.write("Minterms (rows where F = 1): " + number_list(minterms))
                    for number, rows in enumerate(sop_columns, start=1):
                        st.write("**Column " + str(number) + "**")
                        st.dataframe(pd.DataFrame(rows), hide_index=True)

            with tab_pos:
                if len(maxterms) == 0:
                    st.write("The expression is always 1, so F = 1.")
                else:
                    st.write("Maxterms (rows where F = 0): " + number_list(maxterms))
                    st.caption("For POS the zeros of F are simplified. In each sum term, 0 gives the plain variable and 1 gives the complemented variable.")
                    for number, rows in enumerate(pos_columns, start=1):
                        st.write("**Column " + str(number) + "**")
                        st.dataframe(pd.DataFrame(rows), hide_index=True)

    except Exception as error:
        st.error("Invalid expression: " + str(error))
