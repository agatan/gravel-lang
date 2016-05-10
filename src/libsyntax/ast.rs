use pos::{Pos, HasPos};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Name(pub u32);

#[derive(Debug)]
pub struct WithPos<T> {
    pub node: T,
    pub position: Pos,
}

impl<T> HasPos for WithPos<T> {
    fn pos(&self) -> &Pos {
        &self.position
    }

    fn with_pos(self, pos: Pos) -> WithPos<T> {
        WithPos { position: pos, ..self }
    }
}

impl<T> WithPos<T> {
    pub fn new(node: T, pos: Pos) -> WithPos<T> {
        WithPos {
            node: node,
            position: pos,
        }
    }
}

// Expressions

pub type Expr = WithPos<ExprNode>;

#[derive(Debug)]
pub enum ExprNode {
    IntLit(i64),
    BoolLit(bool),
    Var(Name),
    Binary(BinopData),
    Unary(UnaryData),
    Parens(Box<Expr>),
    Call(CallData),
    Construct(ConstructData),
    Block(BlockData),
}

#[derive(Debug)]
pub struct BinopData {
    pub op: Name,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

#[derive(Debug)]
pub struct UnaryData {
    pub op: Name,
    pub operand: Box<Expr>,
}

#[derive(Debug)]
pub struct CallData {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}

#[derive(Debug)]
pub struct ConstructData {
    pub name: Name,
    pub values: Vec<(Name, Box<Expr>)>,
}

#[derive(Debug)]
pub struct BlockData {
    pub stmts: Vec<Stmt>,
    pub last_expr: Option<Box<Expr>>,
}

// Statements
pub type Stmt = WithPos<StmtNode>;

#[derive(Debug)]
pub enum StmtNode {
    Expr(Expr),
    Def(Def),
    Return(Option<Expr>),
}

// Definitions
pub type Def = WithPos<DefNode>;

#[derive(Debug)]
pub enum DefNode {
    Let(LetData),
    Func(FuncData),
    Generic(GenericData),
}

#[derive(Debug)]
pub struct LetData {
    pub sym: Name,
    pub typ: Option<Type>,
    pub value: Expr,
}

#[derive(Debug)]
pub struct FuncData {
    pub name: Name,
    pub params: Vec<(Name, Type)>,
    pub ret: Option<Type>,
    pub body: Expr,
}

#[derive(Debug)]
pub struct GenericData {
    pub constraints: Vec<Constraint>,
    pub def: Box<Def>,
}

#[derive(Debug)]
pub struct Constraint {
    pub type_var: Type,
    pub traits: Vec<Type>,
}

// Types
pub type Type = WithPos<TypeNode>;

#[derive(Debug)]
pub enum TypeNode {
    Primary(Name), // int, bool, SomeStruct, SomeEnum
    Instantiate(Box<Type>, Vec<Type>), // List!(int), Option!(bool)
    FuncPtr(FuncPtrData), // function(int, bool): String
}

#[derive(Debug)]
pub struct InstantiateData {
    pub base: Box<Type>,
    pub args: Vec<Type>,
}

#[derive(Debug)]
pub struct FuncPtrData {
    pub params: Vec<Type>,
    pub ret: Box<Type>,
}
