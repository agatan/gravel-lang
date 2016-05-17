extern crate combine;
extern crate combine_language;

pub mod util;
pub mod ast;
pub mod pos;
pub mod parse;
pub mod error;

use util::interner::StrInterner;
use parse::Context;

use combine::Parser;
use std::rc::Rc;

pub fn parse_module<'a>(interner: &'a StrInterner,
                        file_name: Rc<String>,
                        src: &'a str)
                        -> Result<ast::Module, error::Error<&'a str>> {

    let ctx = Context::<'a, &str>::new(file_name, interner);
    let mut parser = combine::env_parser(&ctx, Context::module);
    parser.parse(src).map(|(module, _)| module)
}
