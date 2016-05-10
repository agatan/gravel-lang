use util::interner::StrInterner;
use pos::Pos;
use ast::Name;

use std::rc::Rc;

use combine::primitives::{ParseResult, State, Stream, SourcePosition};
use combine::char::{string, letter, alpha_num};
use combine::{Parser, ParserExt, satisfy};
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
                    reserved: ["if", "then", "else", "let", "in", "type"]
                                  .iter()
                                  .map(|x| (*x).into())
                                  .collect(),
                },
                op: Identifier {
                    start: satisfy(|c| "+-*/".chars().any(|x| x == c)),
                    rest: satisfy(|c| "+-*/".chars().any(|x| x == c)),
                    reserved: ["+", "-", "*", "/"].iter().map(|x| (*x).into()).collect(),
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

    pub fn integer(&self, input: State<I>) -> ParseResult<i64, I> {
        self.lang_env.integer().parse_state(input)
    }

    pub fn id(&self, input: State<I>) -> ParseResult<Name, I> {
        self.lang_env.identifier().map(|i| self.intern(&i)).parse_state(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use util::interner::{RcStr, StrInterner};
    use ast::Name;

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

        let result = env_parser(&ctx, Context::id).parse("abc");
        assert_eq!(result, Ok((Name(0), "")));
        assert_eq!(interner.get(Name(0)), RcStr::new("abc"));
    }
}
