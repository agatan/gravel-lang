use util::interner::StrInterner;
use pos::Pos;
use ast::{self, Name, Expr, ExprNode, Type};

use std::rc::Rc;

use combine::primitives::{Consumed, ParseResult, State, Stream, SourcePosition};
use combine::char::{string, letter, alpha_num};
use combine::combinator::{Map, And, EnvParser};
use combine::{Parser, ParserExt, between, satisfy, token, env_parser, sep_end_by, many, try};
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
                    reserved: ["=>", ":", "="].iter().map(|x| (*x).into()).collect(),
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

    pub fn ident(&self, input: State<I>) -> ParseResult<Name, I> {
        self.env.identifier().map(|i| self.intern(&i)).parse_state(input)
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

    pub fn construct(&self, input: State<I>) -> ParseResult<Expr, I> {
        let param = self.env
                        .lex(env_parser(self, Context::<'a, I>::ident))
                        .skip(self.env.lex(token(':')))
                        .and(env_parser(self, Context::<'a, I>::expression));
        let params = sep_end_by(self.env.lex(param), self.env.lex(token(',')));
        let construct_parser = (self.env.lex(env_parser(self, Context::<'a, I>::ident)),
                                self.env.braces(params));
        self.with_pos(construct_parser.map(|(name, params)| {
                ExprNode::Construct(ast::ConstructData {
                    name: name,
                    values: params,
                })
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

    pub fn postfix_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let args = self.env
                       .parens(sep_end_by(self.env
                                              .lex(env_parser(self,
                                                              Context::<'a, I>::expression)),
                                          self.env.lex(token(','))));
        let args = self.with_pos(args);
        let ((mut func, args_vec), input) = try!((self.env
                                                  .lex(env_parser(self,
                                                                  Context::<'a, I>::primary_expr)),
                                              many::<Vec<_>, _>(self.env.lex(args)))
                                                 .parse_state(input));
        for args_with_pos in args_vec {
            let pos = args_with_pos.position;
            let calldata = ast::CallData {
                callee: Box::new(func),
                args: args_with_pos.node,
            };
            func = Expr::new(ExprNode::Call(calldata), pos);
        }
        Ok((func, input))
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
        expression_parser(env_parser(self, Context::<'a, I>::postfix_expr),
                          try(env_parser(self, Context::<'a, I>::operator)),
                          mk_binop)
            .parse_state(input)
    }

    pub fn expression(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.env.lex(env_parser(self, Context::<'a, I>::binary_expr)).parse_state(input)
    }

    // type
    pub fn parse_type(&self, input: State<I>) -> ParseResult<Type, I> {
        unimplemented!()
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
        '=' | '!' => 5,
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
    }

    #[test]
    fn postfix_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::expression);

        assert_eq!(parser.parse("hoge (1, 2, 3) ( f() )").unwrap().1, "");
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
}
