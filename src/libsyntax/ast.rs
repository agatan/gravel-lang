use pos::Pos;

#[derive(Clone, Copy, Debug)]
pub struct Name(pub u32);

#[derive(Debug)]
pub struct Expr {
    pub node: ExprNode,
    pub position: Pos,
}

impl Expr {
    fn new(node: ExprNode, pos: Pos) -> Expr {
        Expr {
            node: node,
            position: pos,
        }
    }
}

#[derive(Debug)]
pub enum ExprNode {
    IntLit(i64),
    BoolLit(bool),
}
