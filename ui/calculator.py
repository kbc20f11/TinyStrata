"""Small, dependency-free decimal arithmetic for the browser chat's tools.

This accepts an arithmetic expression, not Python code. In particular, no
names, function calls, attribute access, collections or executable statements
are evaluated. Bounds apply before parsing and after every arithmetic step.
"""
from __future__ import annotations

import ast
from decimal import (Context, Decimal, DecimalException, Inexact, Subnormal,
                     Underflow, localcontext)
import re

PRECISION = 100
MAX_EXPRESSION = 2048
MAX_NODES = 128
MAX_DEPTH = 32
MAX_DIGITS = 100
MAX_EXPONENT = 1000
_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")


def calculate(expression):
    """Return bounded base-10 arithmetic and disclose any precision rounding.

    ``//`` and ``%`` use Python's floor and divisor-sign conventions. Powers
    accept integer exponents only. Nonterminating division is rounded to 100
    significant decimal digits; it is never presented as exact.
    """
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_EXPRESSION:
        raise ValueError(f"Expected an arithmetic expression of at most {MAX_EXPRESSION} characters")
    expression = expression.strip()
    if not expression or "\n" in expression or "\r" in expression or "#" in expression:
        raise ValueError("Expected one arithmetic expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("Invalid arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError("Arithmetic expression has too many operations")

    def bounded(value):
        if not value.is_finite() or (value and not -MAX_EXPONENT <= value.adjusted() <= MAX_EXPONENT):
            raise ValueError("Arithmetic result exceeds the supported magnitude")
        return value

    with localcontext(Context(prec=PRECISION, Emax=MAX_EXPONENT, Emin=-MAX_EXPONENT)) as context:
        context.traps[Subnormal] = context.traps[Underflow] = True
        context.clear_flags()

        def visit(node, depth=0):
            if depth > MAX_DEPTH:
                raise ValueError("Arithmetic expression is nested too deeply")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                # Parse the original spelling: AST float values would already
                # have lost decimal precision (or overflowed) before this point.
                spelling = ast.get_source_segment(expression, node)
                match = _NUMBER.fullmatch(spelling or "")
                if not match or (match[1] and abs(int(match[1])) > MAX_EXPONENT):
                    raise ValueError("Use decimal numbers with exponents from -1000 to 1000")
                number = Decimal(spelling)
                if len(number.as_tuple().digits) > MAX_DIGITS or abs(number.as_tuple().exponent) > MAX_EXPONENT:
                    raise ValueError("A number exceeds the supported precision or exponent")
                return bounded(number)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand, depth + 1)
                return value if isinstance(node.op, ast.UAdd) else value.copy_negate()
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                raise ValueError("Only decimal numbers, parentheses and + - * / // % ** are supported")
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                # Decimal's native // truncates toward zero. Use the exact
                # rational integer quotient instead, so negative operands obey
                # ordinary Python arithmetic without a rounded division first.
                ln, ld = left.as_integer_ratio()
                rn, rd = right.as_integer_ratio()
                quotient = (ln * rd) // (ld * rn)
                # Compute the exact remainder's numerator before decimal
                # division; left - quotient * right would suffer cancellation.
                value = Decimal(quotient) if isinstance(node.op, ast.FloorDiv) else Decimal(ln * rd - quotient * ld * rn) / Decimal(ld * rd)
            else:
                if right != right.to_integral_value() or abs(right) > MAX_EXPONENT:
                    raise ValueError("A power requires an integer exponent from -1000 to 1000")
                value = left ** int(right)
            return bounded(value)

        try:
            result = visit(tree.body)
        except (DecimalException, ZeroDivisionError, OverflowError):
            raise ValueError("Arithmetic is undefined or exceeds the supported limits") from None
        # Fixed-point strings preserve large integers and decimal values without
        # JSON's binary float conversion. Magnitude bounds keep this small.
        text = "0" if not result else format(result, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return {"expression": expression, "result": text,
                "precision": PRECISION, "rounded": bool(context.flags[Inexact])}
