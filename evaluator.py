import collections
import contextlib
import itertools
from inspect import Signature, Parameter

import utils
import cstnodes as C
import typechecker
import typednodes as T
from environments import Namespace, Object, add_namespace

class FunctionObject:
    def __init__(self, f, name=None):
        self.f = f
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return f'<{self.__class__.__qualname__} object {self.name} at 0x{id(self):x}>'
        return super().__repr__()

    def __call__(self, *args, **kwds):
        return self.f(*args, **kwds)

class Tuple:
    def __init__(self, t):
        self.t = tuple(t)

    def __repr__(self):
        return f'Tuple({self.t!r})'

    def __str__(self):
        return '(' + ', '.join(map(str, self.t)) + ')'

    def __eq__(self, other):
        if isinstance(other, Tagged):
            return self.tag == other.tag and self.value == other.value

class Tagged:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

    def __repr__(self):
        return f'Tagged({self.tag!r}, {self.value!r})'

    def __str__(self):
        if self.value is None:
            return f'!{self.tag}'
        return f'!{self.tag} {self.value}'

    def __eq__(self, other):
        if isinstance(other, Tagged):
            return self.tag == other.tag and self.value == other.value

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

FakeToken = collections.namedtuple('FakeToken', 'type string pos')
def gensym(fresh_vars=itertools.count(0)):
    return C.IdExpression(FakeToken('id', 'G' + str(next(fresh_vars)), pos=[]))

def stringify(expr):
    return C.StringExpression([FakeToken('string', utils.to_sexpr(expr), pos=expr.pos)])

def my_print(*args):
    print(*args)

def my_str(x):
    return str(x)

def my_eval(expr):
    return evaluate(typechecker.infer(expr))

def my_len(x):
    if isinstance(x, C.ControlStructureLinkExpression):
        return len(x.components)
    return len(x)

DEFAULTS = {
    'stringify': stringify,
    'something': Something,
    'nothing': Nothing,
    'len': my_len,
    'print': my_print,
    'str': my_str,
    'eval': my_eval,
    'gensym': gensym,
    'RuntimeError': RuntimeError,
    'AssertionError': AssertionError,
}

class EVAL_Exception(BaseException):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class EVAL_ControlFlow(BaseException):
    def __init__(self, pos):
        self.pos = pos

class EVAL_EarlyReturn(EVAL_ControlFlow):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class EVAL_Break(EVAL_ControlFlow):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class EVAL_Continue(EVAL_ControlFlow):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class Evaluate:
    """The evaluation context"""

    def __init__(self):
        self.env = Namespace(DEFAULTS)

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
    def clean_subenv(self, new):
        with self.env.clean_subenv():
            self.env.update(new)
            yield

    @contextlib.contextmanager
    def subenv(self, new):
        with self.env.subenv():
            self.env.update(new)
            yield

    @utils.overloadmethod(error_function=assign_error)
    def assign(self, ast, value):
        ...

    @assign.on(T.Tuple)
    def assign_Tuple(self, ast, value):
        if isinstance(value, C.ControlStructureLinkExpression):
            value = value.params + [value.body]
        else:
            value = value.t
        for k, v in zip(ast.exprs, value):
            self.assign(k, v)

    @assign.on(T.Tagged)
    def assign_Tagged(self, ast, value):
        #print(ast, value)
        if value.tag == ast.tag:
            self.assign(ast.expr, value.value)
        else:
            raise EVAL_Exception(RuntimeError(f'Cannot assign !{value.tag} to !{ast.tag}'), pos=None)

    @assign.on(T.Id)
    def assign_Id(self, ast, value):
        self.assign(ast.name, value)

    @assign.on(str)
    def assign_str(self, ast, value):
        self.env[ast] = value

    @utils.overloadmethod(use_as_default=True, error_function=eval_error)
    def eval(self, *args, **kwds):
        try:
            ast, pos = args
        except Exception as e:
            try:
                ast, = args
            except Exception as e:
                raise utils.ApeInternalError(pos=None, msg='Cannot evaluate {args}')
            raise utils.ApeNotImplementedError(pos=None, msg=f'Cannot evaluate {ast.__class__.__name__}')
        raise utils.ApeNotImplementedError(pos=pos, msg=f'Cannot evaluate {ast.__class__.__name__}')

    @eval.wrapper()
    def eval_wrapper(self, old, new):
        #if not isinstance(old, utils.Ast):
        #    print('EVAL', old, new)
        return new

    @eval.on(utils.Ast)
    def eval_Ast(self, ast):
        if type(ast.node) == tuple:
            raise utils.ApeInternalError(ast.pos, msg='This should never be a tuple, error somewhere else')
        return self.eval(ast.node, ast.pos)

    def recursive_eval(self, args, pos):
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
        if value is True:
            return C.TrueExpression(pos)
        if value is False:
            return C.FalseExpression(pos)
        if value == []:
            return C.EmptyListExpression(pos)
        if value == set():
            return C.EmptySetExpression(pos)
        if value == {}:
            return C.EmptyDictExpression(pos)
        if value == ():
            return C.EmptyTupleExpression(pos)
        if isinstance(value, Tuple):
            return self.unevaluate(value.t, pos)
        if type(value) is tuple:
            return C.TupleLiteral([self.unevaluate(v, pos) for v in value], pos)
        if type(value) is set:
            return C.SetLiteral([self.unevaluate(v, pos) for v in value], pos)
        if type(value) is list:
            return C.ListLiteral([self.unevaluate(v, pos) for v in value], pos)
        if type(value) is int:
            return C.IntExpression.from_decimal(value, pos)  # (FakeToken('decimal_int', value, pos))
        if type(value) is str:
            return C.StringExpression.from_string(FakeToken('string', value, pos), pos)  # ([FakeToken('string', value, pos)])
        if isinstance(value, utils.Expression):
            return value
        raise NotImplementedError(f'Have not implemented "{value!r}" in unevaluator')

    def add_namespace(self, name, env):
        environment = self.env
        for part in name.parts[:-1]:
            environment[part.name] = Namespace({})
            environment = environment[part.name]
        environment[name.parts[-1].name] = Namespace(env)

    @eval.on(T.NamespaceDefn)
    def eval_NamespaceDefn(self, ast, pos):
        self.env.printout()
        with self.clean_subenv({}):
            self.eval(ast.expr)
            env = self.env.env.maps[0]
        add_namespace(self.env, ast.name, env)

    @eval.on(T.Quote)
    def eval_Quote(self, ast, pos):
        if isinstance(ast.cls, C.Expression):
            return ast.cls
        args = tuple(self.recursive_eval(a, pos) for a in ast.args)
        return ast.cls(*args)

    @eval.on(T.Tagged)
    def eval_Tagged(self, ast, pos):
        if ast.expr is None:
            return Tagged(ast.tag, None)
        return Tagged(ast.tag, self.eval(ast.expr))

    @eval.on(T.Comparison)
    def eval_Comparison(self, ast, pos):
        va = self.eval(ast.a)
        vb = self.eval(ast.b)
        #print('va', va, repr(va))
        #print('vb', vb, repr(vb))
        if ast.op == 'eq':
            return va == vb
        if ast.op == 'ne':
            return not (va == vb)
        self.raise_op(ast, ast.op, pos)

    @eval.on(T.SetLit)
    def eval_SetLit(self, ast, pos):
        return {self.eval(expr) for expr in ast.exprs}

    @eval.on(T.EmptySet)
    def eval_EmptySet(self, ast, pos):
        return set()

    @eval.on(T.EmptyList)
    def eval_EmptyList(self, ast, pos):
        return []

    @eval.on(T.List)
    def eval_List(self, ast, pos):
        return [self.eval(expr) for expr in ast.exprs]

    @eval.on(T.LogicalAnd)
    def eval_LogicalAnd(self, ast, pos):
        return all(self.eval(expr) for expr in ast.exprs)

    @eval.on(T.LogicalOr)
    def eval_LogicalOr(self, ast, pos):
        return any(self.eval(expr) for expr in ast.exprs)

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

    @eval.on(T.While)
    def eval_While(self, ast, pos):
        try:
            while self.eval(ast.cond):
                try:
                    self.eval(ast.body)
                except EVAL_Continue:
                    pass
        except EVAL_Break:
            pass
        else:
            if ast.alt is not None:
                self.eval(ast.alt)
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

    @eval.on(T.Index)
    def eval_Index(self, ast, pos):
        value = self.eval(ast.expr)
        for idx in ast.indices:
            #print('value', value)
            value = value[self.eval(idx)]
        return value

    @eval.on(T.Call)
    def eval_Call(self, ast, pos):
        f = self.eval(ast.f)
        args = [self.eval(a) for a in ast.args]
        result = f(*args)
        return result

    @eval.on(T.Tuple)
    def eval_Tuple(self, ast, pos):
        results = []
        for expr in ast.exprs:
            results.append(self.eval(expr))
        if len(results) == 1:
            return results[0]
        return Tuple(results)

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
            # print('ASSIGN', t, v)
            self.assign(t, v)

    @eval.on(T.Yield)
    def eval_Yield(self, ast, pos):
        raise utils.ApeNotImplementedError(pos, 'Wait a minute')

    @eval.on(T.Pass)
    def eval_Pass(self, ast, pos):
        return

    def eval_function_body(self, sig, body, pos, name=None):
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

        if name is None:
            name = '<lambda>'

        return FunctionObject(do_evaluate, name)

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
                    # Default values for parameters are evaluated when the
                    # function is defined, just like in Python
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
        self.assign(ast.name, self.eval_function_body(sig, ast.body, pos, name=ast.name))

def evaluate(ast):
    try:
        return Evaluate().eval(ast)
    except EVAL_EarlyReturn as exc:
        raise utils.ApeEvaluationError(pos=exc.pos, msg='Cannot return when not inside function')
    except EVAL_Exception as exc:
        raise utils.ApeEvaluationError(pos=exc.pos, msg=f'Unhandled APE exception: {repr(exc.value)}')
    except utils.ApeError as exc:
        raise utils.ApeEvaluationError(pos=None, msg='Unexpected error while evaluating') from exc
