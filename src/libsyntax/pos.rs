use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pos {
    file_name: Rc<String>,
    line: i32,
    column: i32,
}

pub trait HasPos {
    fn pos(&self) -> &Pos;
    fn with_pos(self, pos: Pos) -> Self;
}
