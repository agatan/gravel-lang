use util::interner::StrInterner;
use pos::Pos;
use ast::{self, Name, Expr, ExprNode, Type};

use std::rc::Rc;

use combine::primitives::{Consumed, ParseResult, State, Stream, SourcePosition};
use combine::char::{string, letter, alpha_num};
use combine::combinator::{Map, And, EnvParser};
use combine::{Parser, ParserExt, between, satisfy, token, env_parser, sep_end_by, many};
use combine_language::{LanguageEnv, LanguageDef, Identifier};

pub struct Context<'a, I> {
    file_name: Rc<String>,
    interner: &'a StrInterner,
    lang_env: LanguageEnv<'a, I>,
}

impl<'a, I> Context<'a, I>
    where I: Stream<Item = char>
{
    pub fn new(file_name: Rc<String>, interner: &StrInterner) -> Context<&str> {
        Context {
            file_name: file_name,
            interner: interner,
            lang_env: LanguageEnv::new(LanguageDef {
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
        self.lang_env.identifier().map(|i| self.intern(&i)).parse_state(input)
    }

    // expression
    pub fn integer(&self, input: State<I>) -> ParseResult<i64, I> {
        self.lang_env.integer().parse_state(input)
    }

    pub fn boolean(&self, input: State<I>) -> ParseResult<bool, I> {
        self.lang_env
            .reserved("true")
            .map(|_| true)
            .or(self.lang_env.reserved("false").map(|_| false))
            .parse_state(input)
    }

    pub fn primary_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let boolean = env_parser(self, Context::<'a, I>::boolean).map(|t| ExprNode::BoolLit(t));
        let integer = env_parser(self, Context::<'a, I>::integer).map(|v| ExprNode::IntLit(v));
        let var = env_parser(self, Context::<'a, I>::ident).map(|name| ExprNode::Var(name));
        let parens = between(self.lang_env.lex(token('(')),
                             self.lang_env.lex(token(')')),
                             env_parser(self, Context::<'a, I>::expression));
        self.with_pos(boolean.or(integer).or(var)).or(parens).parse_state(input)
    }

    pub fn postfix_expr(&self, input: State<I>) -> ParseResult<Expr, I> {
        let args = self.lang_env
                       .parens(sep_end_by(self.lang_env
                                              .lex(env_parser(self,
                                                              Context::<'a, I>::expression)),
                                          self.lang_env.lex(token(','))));
        let args = self.with_pos(args);
        let ((mut func, args_vec), input) = try!((self.lang_env
                                                  .lex(env_parser(self,
                                                                  Context::<'a, I>::primary_expr)),
                                              many::<Vec<_>, _>(self.lang_env.lex(args)))
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

    pub fn expression(&self, input: State<I>) -> ParseResult<Expr, I> {
        self.lang_env.lex(env_parser(self, Context::<'a, I>::postfix_expr)).parse_state(input)
    }

    // type
    pub fn parse_type(&self, input: State<I>) -> ParseResult<Type, I> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use util::interner::{RcStr, StrInterner};
    use ast::*;

    use std::rc::Rc;

    use combine::*;

    #[test]
    fn integer() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);

        let result = env_parser(&ctx, Context::integer).parse("123");
        assert_eq!(result, Ok((123, "")));
    }

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
    fn boolean() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);

        let result = env_parser(&ctx, Context::boolean).parse("true");
        assert_eq!(result, Ok((true, "")));
    }

    #[test]
    fn primary_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::primary_expr);

        assert_eq!(parser.parse("true").unwrap().0.node,
                   ExprNode::BoolLit(true));
        assert_eq!(parser.parse("123").unwrap().0.node, ExprNode::IntLit(123));
    }

    #[test]
    fn postfix_expr() {
        let interner = StrInterner::new();
        let file = Rc::new("test".to_owned());
        let ctx = Context::<&str>::new(file, &interner);
        let mut parser = env_parser(&ctx, Context::postfix_expr);

        assert_eq!(parser.parse("hoge (1, 2, 3) ( f() )").unwrap().1, "");
    }
}
