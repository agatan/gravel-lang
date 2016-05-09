use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pos {
    file_name: Rc<String>,
    line: i32,
    column: i32,
}
