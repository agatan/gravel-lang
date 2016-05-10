use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pos {
    pub file_name: Rc<String>,
    pub line: i32,
    pub column: i32,
}

pub trait HasPos {
    fn pos(&self) -> &Pos;
    fn with_pos(self, pos: Pos) -> Self;
}
