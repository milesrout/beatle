import collections
import contextlib
import itertools
from inspect import Signature, Parameter

import utils
import cstnodes as C
import typednodes as T

class Expression:
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

    def __repr__(self):
        return f'Expression({self.expr})'

class NothingType:
    def __repr__(self):
        return f'nothing'
Nothing = NothingType()

class Something:
    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f'something({self.v})'

FakeToken = collections.namedtuple('FakeToken', 'string pos')
def gensym(fresh_vars=itertools.count(0)):
    return C.IdExpression(FakeToken('G' + str(next(fresh_vars)), pos=[]))

def stringify(expr):
    print('--------------------------------------')
    print(expr, type(expr))
    return C.StringExpression([FakeToken(utils.to_sexpr(expr), pos=expr.pos)])

DEFAULTS = {
    'stringify': stringify,
    'something': Something,
    'nothing': Nothing,
    'print': print,
    'str': str,
    'gensym': gensym,
    'RuntimeError': RuntimeError,
    'AssertionError': AssertionError,
}

class EVAL_Exception(BaseException):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class EVAL_EarlyReturn(BaseException):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class Evaluate:
    """The evaluation context"""

    def __init__(self):
        self.env = collections.ChainMap(DEFAULTS)

    def assign_error(self, ast, value):
        try:
            return utils.ApeInternalError(
                pos=ast.pos,
                msg='Evaluate.assign: no overload found for {}'.format(ast.__class__))
        except Exception:
            return utils.ApeInternalError(
                pos=None,
                msg='Evaluate.assign: no overload found for {}'.format(ast.__class__))

    def eval_error(self, ast, pos):
        try:
            return utils.ApeInternalError(
                pos=pos,
                msg='Evaluate.eval: no overload found for {}'.format(ast.__class__))
        except Exception:
            return utils.ApeInternalError(
                pos=None,
                msg='Evaluate.eval: no overload found for {}'.format(ast.__class__))

    def raise_op(self, ast, op, pos):
        raise utils.ApeInternalError(
            pos=pos,
            msg='Evaluate.eval: no overload found for {} [op={}]'.format(ast.__class__, op))

    @contextlib.contextmanager
    def subenv(self, new):
        old = self.env
        self.env = self.env.new_child(new)
        yield
        self.env = old

    @utils.overloadmethod(error_function=assign_error)
    def assign(self, ast, value):
        ...

    @assign.on(T.Id)
    def assign_IdExpression(self, ast, value):
        self.assign(ast.name, value)

    @assign.on(str)
    def assign_str(self, ast, value):
        self.env[ast] = value

    @utils.overloadmethod(use_as_default=True, error_function=eval_error)
    def eval(self, *args, **kwds):
        #print('args', args)
        #print('kwds', kwds)
        try:
            ast, pos = args
        except Exception as e:
            try:
                ast, = args
            except Exception as e:
                raise utils.ApeInternalError(pos=None, msg='Cannot evaluate {args}')
            raise utils.ApeNotImplementedError(pos=None, msg=f'Cannot evaluate {ast.__class__.__name__}')
        raise utils.ApeNotImplementedError(pos=pos, msg=f'Cannot evaluate {ast.__class__.__name__}')

    @eval.on(utils.Ast)
    def eval_Ast(self, ast):
        # print('-----ast', type(ast), ast)
        if type(ast.node) == tuple:
            print(ast.node)
            raise
        return self.eval(ast.node, ast.pos)

    def recursive_eval(self, args, pos):
        # print('args', type(args), args, pos)
        if isinstance(args, list):
            return list(self.recursive_eval(a, pos) for a in args)
        if type(args) == tuple:
            return tuple(self.recursive_eval(a, pos) for a in args)
        if args.__class__.__name__ == 'Ast':
            return self.unevaluate(self.eval(args), pos)
        if type(args) == T.Quote:
            return self.eval(args, pos)
        return args

    def unevaluate(self, value, pos):
        print('-----value', value, repr(value), type(value), value.__class__)
        if value is True:
            return C.TrueExpression(pos=pos)
        if type(value) is str:
            return C.StringExpression([FakeToken(value, pos)])
        if isinstance(value, utils.Expression):
            return value
            return Expression(value, pos)
        raise NotImplementedError(f'Have not implemented "{value!r}"')

    @eval.on(T.Quote)
    def eval_Quote(self, ast, pos):
        if isinstance(ast.cls, C.Expression):
            return ast.cls
        # print('-----unevaluated', ast.cls, ast.args)
        args = tuple(self.recursive_eval(a, pos) for a in ast.args)
        # print('-----evaluated', ast.cls, args)
        return ast.cls(*args)

    @eval.on(T.Comparison)
    def eval_Comparison(self, ast, pos):
        if ast.op == 'eq':
            return self.eval(ast.a) == self.eval(ast.b)
        self.raise_op(ast, ast.op, pos)

    @eval.on(T.LogicalNot)
    def eval_LogicalNot(self, ast, pos):
        return not self.eval(ast.expr)

    @eval.on(T.IfElifElse)
    def eval_IfElifElse(self, ast, pos):
        if self.eval(ast.if_branch.cond):
            return self.eval(ast.if_branch.body)
        for eib in ast.elif_branches:
            if self.eval(eib.cond):
                return self.eval(eib.body)
        if ast.else_branch is not None:
            return self.eval(ast.else_branch.body)
        return None

    @eval.on(T.Statements)
    def eval_Statements(self, ast, pos):
        value = None
        for stmt in ast.stmts:
            value = self.eval(stmt)
        return value

    @eval.on(T.Unary)
    def eval_Unary(self, ast, pos):
        e = self.eval(ast.expr)
        if ast.op == 'minus':
            return -e
        self.raise_op(ast, ast.op, pos)

    @eval.on(T.Arith)
    def eval_Arith(self, ast, pos):
        l = self.eval(ast.left)
        r = self.eval(ast.right)
        if ast.op == 'plus':
            return l + r
        if ast.op == 'minus':
            return l - r
        if ast.op == 'asterisk':
            return l * r
        raise utils.ApeNotImplementedError(f'{ast.op} is not supported yet', pos=pos)

    @eval.on(T.Bool)
    def eval_Bool(self, ast, pos):
        return ast.value

    @eval.on(T.NoneExpr)
    def eval_NoneExpr(self, ast, pos):
        return None

    @eval.on(T.String)
    def eval_String(self, ast, pos):
        return ast.string

    @eval.on(T.Int)
    def eval_Int(self, ast, pos):
        return int(ast.value)

    @eval.on(T.Id)
    def eval_Id(self, ast, pos):
        return self.env[ast.name]

    @eval.on(T.Call)
    def eval_Call(self, ast, pos):
        print(ast)
        f = self.eval(ast.f)
        args = [self.eval(a) for a in ast.args]
        return f(*args)

    @eval.on(T.Tuple)
    def eval_Tuple(self, ast, pos):
        results = []
        for expr in ast.exprs:
            results.append(self.eval(expr))
        if len(results) == 1:
            return results[0]
        return tuple(results)

    @eval.on(T.Raise)
    def eval_Raise(self, ast, pos):
        v = self.eval(ast.expr)
        raise EVAL_Exception(v, pos)

    @eval.on(T.Return)
    def eval_Return(self, ast, pos):
        v = self.eval(ast.expr)
        raise EVAL_EarlyReturn(v, pos)

    @eval.on(T.Assignment)
    def eval_Assignment(self, ast, pos):
        v = self.eval(ast.expr)
        for t in ast.targets:
            self.assign(t, v)

    @eval.on(T.Yield)
    def eval_Yield(self, ast, pos):
        raise utils.ApeNotImplementedError(pos, 'Wait a minute')

    def eval_function_body(self, sig, body, pos):
        def do_evaluate(*args):
            try:
                ba = sig.bind(*args)
            except TypeError as exc:
                raise utils.ApeEvaluationError(pos=pos, msg='Invalid arguments to function') from exc
            ba.apply_defaults()
            with self.subenv(ba.arguments):
                try:
                    return self.eval(body)
                except EVAL_EarlyReturn as exc:
                    return exc.value
        return do_evaluate

    def eval_parameters(self, ast_params, defaults):
        no_more_pos_params = False
        params = []
        for param, default_ast in zip(ast_params, defaults):
            default = Parameter.empty
            if isinstance(param, C.Param):
                if no_more_pos_params:
                    kind = Parameter.KEYWORD_ONLY
                else:
                    kind = Parameter.POSITIONAL_OR_KEYWORD
                if param.default is not None:
                    default = self.eval(default_ast)
            elif isinstance(param, C.StarVarParams):
                kind = Parameter.VAR_POSITIONAL
            elif isinstance(param, C.StarStarKwParams):
                kind = Parameter.VAR_KEYWORD
            elif isinstance(param, C.EndOfPosParams):
                no_more_pos_params = True
                continue
            params.append(Parameter(param.name.name, kind, default=default))
        return Signature(params)

    @eval.on(T.Function)
    def eval_Function(self, ast, pos):
        sig = self.eval_parameters(ast.params, ast.defaults)
        return self.eval_function_body(sig, ast.body, pos)

    @eval.on(T.FunctionDefinition)
    def eval_FunctionDefinition(self, ast, pos):
        sig = self.eval_parameters(ast.params, ast.defaults)
        self.assign(ast.name, self.eval_function_body(sig, ast.body, pos))

def evaluate(ast):
    try:
        return Evaluate().eval(ast)
    except EVAL_EarlyReturn as exc:
        raise utils.ApeEvaluationError(pos=exc.pos, msg='Cannot return when not inside function')
    except EVAL_Exception as exc:
        raise utils.ApeEvaluationError(pos=exc.pos, msg=f'Unhandled APE exception: {repr(exc.value)}')
    except utils.ApeError as exc:
        raise utils.ApeEvaluationError(pos=None, msg='Unexpected error while evaluating') from exc
