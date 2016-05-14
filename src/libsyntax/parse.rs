use util::interner::StrInterner;
use pos::Pos;
use ast::{self, Name, Expr, ExprNode, Stmt, StmtNode, Def, DefNode, Type, TypeNode};

use std::rc::Rc;

use combine::primitives::{Consumed, ParseResult, State, Stream, SourcePosition};
use combine::char::{string, letter, alpha_num};
use combine::combinator::{Map, And, EnvParser};
use combine::{Parser, ParserExt, between, satisfy, token, env_parser, parser, sep_end_by, sep_by,
              sep_by1, many, try, optional, value};
use combine_language::{LanguageEnv, LanguageDef, Identifier, expression_parser, Assoc, Fixity};

pub struct Context<'a, I> {
    file_name: Rc<String>,
    interner: &'a StrInterner,
    env: LanguageEnv<'a, I>,
}

impl<'a, I> Context<'a, I>
    where I: Stream<Item = char>
{
    pub fn new(file_name: Rc<String>, interner: &StrInterner) -> Context<&str> {
        Context {
            file_name: file_name,
            interner: interner,
            env: LanguageEnv::new(LanguageDef {
                ident: Identifier {
                    start: letter(),
                    rest: alpha_num(),
                    reserved: ["def",
                               "let",
                               "return",
                               "func",
                               "implement",
                               "trait",
                               "struct",
                               "enum",
                               "true",
                               "false"]
                                  .iter()
                                  .map(|x| (*x).into())
                                  .collect(),
                },
                op: Identifier {
                    start: satisfy(|c| "+-*/~@$%^&:<>?".chars().any(|x| x == c)),
                    rest: satisfy(|c| "+-*/~@$%^&:<>?".chars().any(|x| x == c)),
                    reserved: ["=>", ":", "=", "!"].iter().map(|x| (*x).into()).collect(),
                },
                comment_start: string("/*").map(|_| ()),
                comment_end: string("*/").map(|_| ()),
                comment_line: string("//").map(|_| ()),
            }),
        }
    }

    pub fn convert_pos(&self, spos: &SourcePosition) -> Pos {
        Pos {
            file_name: self.file_name.clone(),
            line: spos.line,
            column: spos.column,
        }
    }

    pub fn intern(&self, val: &str) -> Name {
        self.interner.intern(val)
    }

    // Parser definitions...
    fn get_pos(&self, input: State<I>) -> ParseResult<Pos, I> {
        let pos = self.convert_pos(&input.position);
        Ok((pos, Consumed::Empty(input)))
    }

    fn with_pos<P>(&self,
                   p: P)
                   -> Map<And<EnvParser<&Context<'a, I>, I, Pos>, P>,
                          fn((Pos, <P as Parser>::Output)) -> ast::WithPos<<P as Parser>::Output>>
        where P: Parser<Input = I>
    {
        fn mk_with_pos<T>((pos, val): (Pos, T)) -> ast::WithPos<T> {
            ast::WithPos::new(val, pos)
        }
        env_parser(self, Context::<'a, I>::get_pos)
            .and(p)
            .map(mk_with_pos)
    }

    fn ident_tail(&self, input: State<I>) -> ParseResult<String, I> {
        self.env.lex(many(satisfy(|c: char| c.is_alphanumeric() || c == '_'))).parse_state(input)
    }

    pub fn ident(&self, input: State<I>) -> ParseResult<Name, I> {
        (satisfy(char::is_lowercase), env_parser(self, Context::<'a, I>::ident_tail))
            .map(|(c, tail)| self.intern(&format!("{}{}", c, tail)))
            .parse_state(input)
    }

    pub fn uident(&self, input: State<I>) -> ParseResult<Name, I> {
        (satisfy(char::is_uppercase), env_parser(self, Context::<'a, I>::ident_tail))
            .map(|(c, tail)| self.intern(&format!("{}{}", c, tail)))
            .parse_state(input)
    }

    // expression
    pub fn integer(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.with_pos(self.env.integer().map(|i| ExprNode::IntLit(i))).parse_state(input)
    }

    pub fn boolean(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.with_pos(self.env
                          .reserved("true")
                          .map(|_| ExprNode::BoolLit(true))
                          .or(self.env.reserved("false").map(|_| ExprNode::BoolLit(false))))
            .parse_state(input)
    }

    pub fn var(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.with_pos(env_parser(self, Context::<'a, I>::ident).map(|name| ExprNode::Var(name)))
            .parse_state(input)
    }

    fn construct_args(&self, input: State<I>) -> ParseResult<ConstructArgs, I> {
        let param = self.env
                        .lex(env_parser(self, Context::<'a, I>::ident))
                        .skip(self.env.lex(token(':')))
                        .and(env_parser(self, Context::<'a, I>::expression));
        let params = sep_end_by(self.env.lex(param), self.env.lex(token(',')));
        self.env
            .braces(params)
            .map(ConstructArgs::Record)
            .or(optional(self.env
                             .parens(sep_end_by(env_parser(self, Context::<'a, I>::expression),
                                                self.env.symbol(","))))
                    .map(|args| ConstructArgs::Enum(vec_option_to_vec(args))))
            .parse_state(input)
    }

    pub fn construct(&self, input: State<I>) -> ParseResult<Expr, I> {
        let construct_parser = (self.env.lex(env_parser(self, Context::<'a, I>::uident)),
                                env_parser(self, Context::<'a, I>::construct_args));
        self.with_pos(construct_parser.map(|(name, args)| {
                match args {
                    ConstructArgs::Record(params) => {
                        ExprNode::RecordConstruct(ast::RecordConstructData {
                            name: name,
                            values: params,
                        })
                    }
                    ConstructArgs::Enum(args) => {
                        ExprNode::EnumConstruct(ast::EnumConstructData {
                            name: name,
                            values: args,
                        })
                    }
                }
            }))
            .parse_state(input)
    }

    pub fn primary_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let integer = env_parser(self, Context::<'a, I>::integer);
        let boolean = env_parser(self, Context::<'a, I>::boolean);
        let construct = try(env_parser(self, Context::<'a, I>::construct));
        let var = env_parser(self, Context::<'a, I>::var);
        let parens = between(self.env.lex(token('(')),
                             self.env.lex(token(')')),
                             env_parser(self, Context::<'a, I>::expression));
        boolean.or(integer).or(construct).or(var).or(parens).parse_state(input)
    }

    fn args(&self, input: State<I>) -> ParseResult<Vec<Expr>, I> {
        let mut args = self.env
                           .parens(sep_end_by(self.env
                                                  .lex(env_parser(self,
                                                                  Context::<'a, I>::expression)),
                                              self.env.lex(token(','))));
        args.parse_state(input)
    }

    fn postfix(&self, input: State<I>) -> ParseResult<(Postfix, Pos), I> {
        let func_apply = self.with_pos(env_parser(self, Context::<'a, I>::args))
                             .map(|ast::WithPos { node, position }| {
                                 (Postfix::Call(node), position)
                             });
        let field_or_method = env_parser(self, Context::<'a, I>::get_pos)
                                  .skip(self.env.symbol("."))
                                  .and(env_parser(self, Context::<'a, I>::ident))
                                  .and(optional(env_parser(self, Context::<'a, I>::args)))
                                  .map(|((pos, name), opt_args)| match opt_args {
                                      None => (Postfix::Field(name), pos),
                                      Some(args) => (Postfix::Method(name, args), pos),
                                  });
        func_apply.or(field_or_method).parse_state(input)
    }

    pub fn postfix_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let ((mut base, postfixes), input) = try!((env_parser(self, Context::<'a, I>::primary_expr),
                                                   many::<Vec<_>,
                                                          _>(env_parser(self,
                                                                        Context::<'a, I>::postfix)))
                                                      .parse_state(input));

        for (postfix, pos) in postfixes {
            let node = match postfix {
                Postfix::Call(args) => {
                    ExprNode::Call(ast::CallData {
                        callee: Box::new(base),
                        args: args,
                    })
                }
                Postfix::Field(name) => {
                    ExprNode::RefField(ast::RefFieldData {
                        base: Box::new(base),
                        field: name,
                    })
                }
                Postfix::Method(name, args) => {
                    ExprNode::MethodCall(ast::MethodCallData {
                        reciever: Box::new(base),
                        method: name,
                        args: args,
                    })
                }
            };
            base = Expr::new(node, pos);
        }

        Ok((base, input))
    }

    pub fn block_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let block = self.env
                        .braces((many(try(env_parser(self, Context::<'a, I>::statement))),
                                 optional(env_parser(self, Context::<'a, I>::expression)
                                              .map(Box::new))))
                        .map(|(ss, eopt)| {
                            ExprNode::Block(ast::BlockData {
                                stmts: ss,
                                last_expr: eopt,
                            })
                        });
        self.with_pos(block).parse_state(input)
    }

    fn operator(&self, input: State<I>) -> ParseResult<(((Name, ast::BinopBase), Pos), Assoc), I> {
        self.with_pos(self.env.op())
            .map(|op_with_pos| {
                let name = self.intern(&op_with_pos.node);
                let (base, assoc) = op_assoc(&op_with_pos.node);
                (((name, base), op_with_pos.position), assoc)
            })
            .parse_state(input)
    }

    pub fn binary_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        expression_parser(env_parser(self, Context::<'a, I>::postfix_expr)
                              .or(env_parser(self, Context::<'a, I>::block_expr)),
                          try(env_parser(self, Context::<'a, I>::operator)),
                          mk_binop)
            .parse_state(input)
    }

    pub fn expression(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.env.lex(env_parser(self, Context::<'a, I>::binary_expr)).parse_state(input)
    }

    // statement
    fn expr_stmt(&self, input: State<I>) -> ParseResult<Stmt, I> {
        self.with_pos(env_parser(self, Context::<'a, I>::expression).map(StmtNode::Expr))
            .skip(self.env.lex(token(';')))
            .parse_state(input)
    }

    fn def_stmt(&self, input: State<I>) -> ParseResult<Stmt, I> {
        self.with_pos(env_parser(self, Context::<'a, I>::definition).map(StmtNode::Def))
            .parse_state(input)
    }

    fn return_stmt(&self, input: State<I>) -> ParseResult<Stmt, I> {
        self.with_pos(self.env
                          .reserved("return")
                          .with(optional(env_parser(self, Context::<'a, I>::expression)))
                          .skip(self.env.lex(token(';')))
                          .map(StmtNode::Return))
            .parse_state(input)
    }

    pub fn statement(&self, input: State<I>) -> ParseResult<Stmt, I> {
        try(env_parser(self, Context::<'a, I>::expr_stmt))
            .or(env_parser(self, Context::<'a, I>::def_stmt))
            .or(env_parser(self, Context::<'a, I>::return_stmt))
            .parse_state(input)
    }

    // definition
    fn type_spec(&self, input: State<I>) -> ParseResult<Type, I> {
        self.env
            .reserved_op(":")
            .with(env_parser(self, Context::<'a, I>::parse_type))
            .parse_state(input)
    }
    fn let_def(&self, input: State<I>) -> ParseResult<Def, I> {
        let parser = (self.env.reserved("let"),
                      env_parser(self, Context::<'a, I>::ident),
                      optional(env_parser(self, Context::<'a, I>::type_spec)),
                      self.env.reserved_op("="),
                      env_parser(self, Context::<'a, I>::expression),
                      self.env.symbol(";"));
        let let_def_parser = parser.map(|(_, name, typ, _, value, _)| {
            DefNode::Let(ast::LetData {
                sym: name,
                typ: typ,
                value: value,
            })
        });
        self.with_pos(let_def_parser).parse_state(input)
    }

    // T: ToString + Eq
    fn constraint(&self, input: State<I>) -> ParseResult<ast::Constraint, I> {
        let parser = (env_parser(self, Context::<'a, I>::parse_type),
                      optional(self.env
                                   .reserved_op(":")
                                   .with(sep_by1(env_parser(self, Context::<'a, I>::parse_type),
                                                 self.env.symbol(",")))));
        parser.map(|(type_var, traits)| {
                  ast::Constraint {
                      type_var: type_var,
                      traits: vec_option_to_vec(traits),
                  }
              })
              .parse_state(input)
    }

    fn func_def(&self, input: State<I>) -> ParseResult<Def, I> {
        let param = (env_parser(self, Context::<'a, I>::ident),
                     env_parser(self, Context::<'a, I>::type_spec));
        let opt_generic_params =
            optional(self.env.angles(sep_by1(env_parser(self, Context::<'a, I>::constraint),
                                             self.env.symbol(","))))
                .map(vec_option_to_vec);
        let parser = (self.env.reserved("def"),
                      opt_generic_params,
                      env_parser(self, Context::<'a, I>::ident),
                      self.env.parens(sep_by(param, self.env.symbol(","))),
                      optional(env_parser(self, Context::<'a, I>::type_spec)),
                      self.env.reserved_op("="),
                      env_parser(self, Context::<'a, I>::expression));
        let func_def_parser = parser.map(|(_, generic_params, name, params, ret, _, body)| {
            DefNode::Func(ast::FuncData {
                generic_params: generic_params,
                name: name,
                params: params,
                ret: ret,
                body: body,
            })
        });
        self.with_pos(func_def_parser).parse_state(input)
    }

    fn struct_def(&self, input: State<I>) -> ParseResult<Def, I> {
        let param = (env_parser(self, Context::<'a, I>::ident),
                     env_parser(self, Context::<'a, I>::type_spec));
        let opt_generic_params =
            optional(self.env.angles(sep_by1(env_parser(self, Context::<'a, I>::constraint),
                                             self.env.symbol(","))))
                .map(vec_option_to_vec);
        let parser = (self.env.reserved("struct"),
                      env_parser(self, Context::<'a, I>::uident),
                      opt_generic_params,
                      self.env.braces(many(param.skip(self.env.symbol(";")))));
        self.with_pos(parser.map(|(_, name, gen_params, params)| {
                DefNode::Struct(ast::StructData {
                    generic_params: gen_params,
                    name: name,
                    params: params,
                })
            }))
            .parse_state(input)
    }

    fn enum_def(&self, input: State<I>) -> ParseResult<Def, I> {
        let elem = (env_parser(self, Context::<'a, I>::uident),
                    optional(self.env
                                 .parens(sep_by1(env_parser(self, Context::<'a, I>::parse_type),
                                                 self.env.symbol(","))))
                        .map(vec_option_to_vec))
                       .map(|(name, params)| {
                           ast::EnumElem {
                               name: name,
                               params: params,
                           }
                       });
        let opt_generic_params =
            optional(self.env.angles(sep_by1(env_parser(self, Context::<'a, I>::constraint),
                                             self.env.symbol(","))))
                .map(vec_option_to_vec);
        let parser = (self.env.reserved("enum"),
                      env_parser(self, Context::<'a, I>::uident),
                      opt_generic_params,
                      self.env.braces(many(elem.skip(self.env.symbol(";")))));
        self.with_pos(parser.map(|(_, name, gen_params, elems)| {
                DefNode::Enum(ast::EnumData {
                    name: name,
                    generic_params: gen_params,
                    elems: elems,
                })
            }))
            .parse_state(input)
    }

    pub fn definition(&self, input: State<I>) -> ParseResult<Def, I> {
        env_parser(self, Context::<'a, I>::let_def)
            .or(env_parser(self, Context::<'a, I>::func_def))
            .or(env_parser(self, Context::<'a, I>::struct_def))
            .or(env_parser(self, Context::<'a, I>::enum_def))
            .parse_state(input)
    }

    // type
    fn primary_type(&self, input: State<I>) -> ParseResult<Type, I> {
        self.with_pos(env_parser(self, Context::<'a, I>::uident).map(TypeNode::Primary))
            .parse_state(input)
    }

    fn instantiate_type(&self, input: State<I>) -> ParseResult<Type, I> {
        let inst_arg =
            self.with_pos(self.env
                              .symbol("!")
                              .with(self.env
                                        .angles(sep_by(env_parser(self,
                                                                  Context::<'a, I>::parse_type),
                                                       self.env.symbol(",")))));
        (env_parser(self, Context::<'a, I>::primary_type), many::<Vec<_>, _>(inst_arg))
            .map(|(mut base, args_vec)| {
                for ast::WithPos { node, position } in args_vec {
                    base = Type::new(TypeNode::Instantiate(Box::new(base), node), position)
                }
                base
            })
            .parse_state(input)
    }

    fn funcptr_type(&self, input: State<I>) -> ParseResult<Type, I> {
        let parser = (self.env.reserved("func"),
                      self.env.parens(sep_by(env_parser(self, Context::<'a, I>::parse_type),
                                             self.env.symbol(","))),
                      self.env.reserved_op(":"),
                      env_parser(self, Context::<'a, I>::parse_type));
        self.with_pos(parser.map(|(_, params, _, ret)| {
                TypeNode::FuncPtr(ast::FuncPtrData {
                    params: params,
                    ret: Box::new(ret),
                })
            }))
            .parse_state(input)
    }

    pub fn parse_type(&self, input: State<I>) -> ParseResult<Type, I> {
        env_parser(self, Context::<'a, I>::funcptr_type)
            .or(env_parser(self, Context::<'a, I>::instantiate_type))
            .parse_state(input)
    }
}

fn mk_binop(lhs: Expr, (op, pos): ((Name, ast::BinopBase), Pos), rhs: Expr) -> Expr {
    let binop_data = ast::BinopData {
        op: op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    };
    Expr::new(ExprNode::Binary(binop_data), pos)
}

fn op_assoc(op: &str) -> (ast::BinopBase, Assoc) {
    if op.ends_with("=") {
        return (ast::BinopBase::Right,
                Assoc {
            precedence: 0,
            fixity: Fixity::Right,
        });
    }
    let (base, fixity) = if op.ends_with(':') {
        (ast::BinopBase::Right, Fixity::Right)
    } else {
        (ast::BinopBase::Left, Fixity::Left)
    };

    let prec = match op.chars().next().unwrap() {
        '|' => 1,
        '^' => 2,
        '&' => 3,
        '<' | '>' => 4,
        '=' => 5,
        ':' => 6,
        '+' | '-' => 7,
        '*' | '/' | '%' => 8,
        _ => 9,
    };

    (base,
     Assoc {
        precedence: prec,
        fixity: fixity,
    })
}

fn vec_option_to_vec<T>(v: Option<Vec<T>>) -> Vec<T> {
    match v {
        None => Vec::new(),
        Some(v) => v,
    }
}

enum Postfix {
    Call(Vec<Expr>),
    Field(Name),
    Method(Name, Vec<Expr>),
}

enum ConstructArgs {
    Record(Vec<(Name, Expr)>),
    Enum(Vec<Expr>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use util::interner::{RcStr, StrInterner};
    use ast::*;

    use std::rc::Rc;

    use combine::*;

    #[test]
    fn identifier() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);

        let result = env_parser(&ctx, Context::ident).parse("abc");
        assert_eq!(result, Ok((Name(0), "")));
        assert_eq!(interner.get(Name(0)), RcStr::new("abc"));
    }

    #[test]
    fn primary_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::expression);

        assert_eq!(parser.parse("true").unwrap().0.node,
                   ExprNode::BoolLit(true));
        assert_eq!(parser.parse("123").unwrap().0.node, ExprNode::IntLit(123));
        assert_eq!(parser.parse("Point { x: 123 , y : 456 }").unwrap().1, "");
        assert_eq!(parser.parse("None").unwrap().1, "");
        assert_eq!(parser.parse("Some(1)").unwrap().1, "");
    }

    #[test]
    fn postfix_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::expression);

        assert_eq!(parser.parse("hoge (1, 2, 3) ( f() )").unwrap().1, "");
        assert_eq!(parser.parse("value.method(1, 2)").unwrap().1, "");
        assert_eq!(parser.parse("value.method").unwrap().1, "");
        assert_eq!(parser.parse("value.field.method1()(1, 2).method2()").unwrap().1,
                   "");
    }

    #[test]
    fn block_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::expression);

        assert_eq!(parser.parse("{ 1; hoge(); }").unwrap().1, "");
        assert_eq!(parser.parse("{ 1; hoge(); 3 + f()}").unwrap().1, "");
    }

    #[test]
    fn binary_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::expression);

        let result = parser.parse("1 + 2 * 3 <:> 4");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().1, "");
    }

    #[test]
    fn def_stmt() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::statement);

        assert_eq!(parser.parse("let x = 1 + 2;").unwrap().1, "");
    }

    #[test]
    fn return_stmt() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::statement);

        assert_eq!(parser.parse("return 1 ; ").unwrap().1, "");
    }

    #[test]
    fn let_def() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::definition);

        assert_eq!(parser.parse("let x: Int = 1;").unwrap().1, "");
        assert_eq!(parser.parse("let x = 1;").unwrap().1, "");
    }

    #[test]
    fn func_def() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::definition);

        assert_eq!(parser.parse("def f(x: Int): Int = 1").unwrap().1, "");
        assert_eq!(parser.parse("def f() = g()").unwrap().1, "");
        assert_eq!(parser.parse("def<T: ToString> print(x: T) = g(x)").unwrap().1,
                   "");
    }

    #[test]
    fn struct_def() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::definition);

        assert_eq!(parser.parse("struct Point { x: Int; y: Int; }").unwrap().1,
                   "");
        assert_eq!(parser.parse("struct Pair<T, U> { first: T; second: U; }").unwrap().1,
                   "");
    }

    #[test]
    fn enum_def() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::definition);

        assert_eq!(parser.parse("enum Option<T> { Some(T); None; }").unwrap().1,
                   "");
        assert_eq!(parser.parse("enum Direction { Left; Right; }").unwrap().1,
                   "");
    }

    #[test]
    fn primary_type() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::parse_type);

        assert_eq!(parser.parse("Int").unwrap().1, "");
    }

    #[test]
    fn instantiate_type() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::parse_type);

        assert_eq!(parser.parse("Generic!<Int, Option!<Bool>>").unwrap().1, "");
    }

    #[test]
    fn funcptr_type() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::parse_type);

        assert_eq!(parser.parse("func(Option!<Int>, Bool): String").unwrap().1,
                   "");
    }
}
