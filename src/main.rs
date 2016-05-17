extern crate gravel;
extern crate syntax;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::rc::Rc;

fn main() {
    let path = Path::new("example/sample.gv");
    let display = path.to_string_lossy().into_owned();

    let mut file = File::open(&path).unwrap();

    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();

    let interner = syntax::util::interner::StrInterner::new();
    let file_name = Rc::new(display);

    println!("{:?}",
             syntax::parse_module(&interner, file_name, &s).unwrap());
}
