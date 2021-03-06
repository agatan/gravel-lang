use pos::{Pos, HasPos};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Name(pub u32);

#[derive(Debug, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub enum ExprNode {
    IntLit(i64),
    BoolLit(bool),
    Var(Name),
    Binary(BinopData),
    Unary(UnaryData),
    Parens(Box<Expr>),
    RefField(RefFieldData),
    MethodCall(MethodCallData),
    Call(CallData),
    RecordConstruct(RecordConstructData),
    EnumConstruct(EnumConstructData),
    Block(BlockData),
}

/// binary expression's base direction.
/// `a + b` and `a.+(b)` are same expression and `a + b`'s base is `Left`.
/// `a :: b` and `b.::(a)` are same expression and `a :: b`'s base is `Right`.
#[derive(Debug, PartialEq)]
pub enum BinopBase {
    Right,
    Left,
}

#[derive(Debug, PartialEq)]
pub struct BinopData {
    pub op: (Name, BinopBase),
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct UnaryData {
    pub op: Name,
    pub operand: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct RefFieldData {
    pub base: Box<Expr>,
    pub field: Name,
}

#[derive(Debug, PartialEq)]
pub struct MethodCallData {
    pub reciever: Box<Expr>,
    pub method: Name,
    pub args: Vec<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct CallData {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct RecordConstructData {
    pub name: Name,
    pub values: Vec<(Name, Expr)>,
}

#[derive(Debug, PartialEq)]
pub struct EnumConstructData {
    pub name: Name,
    pub values: Vec<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct BlockData {
    pub stmts: Vec<Stmt>,
    pub last_expr: Option<Box<Expr>>,
}

// Statements
pub type Stmt = WithPos<StmtNode>;

#[derive(Debug, PartialEq)]
pub enum StmtNode {
    Expr(Expr),
    Def(Def),
    Return(Option<Expr>),
}

// Definitions
pub type Def = WithPos<DefNode>;

#[derive(Debug, PartialEq)]
pub enum DefNode {
    Let(LetData),
    Func(FuncData),
    Struct(StructData),
    Enum(EnumData),
}

#[derive(Debug, PartialEq)]
pub struct LetData {
    pub sym: Name,
    pub typ: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, PartialEq)]
pub struct FuncData {
    pub generic_params: Vec<Constraint>,
    pub name: Name,
    pub params: Vec<(Name, Type)>,
    pub ret: Option<Type>,
    pub body: Expr,
}

#[derive(Debug, PartialEq)]
pub struct StructData {
    pub generic_params: Vec<Constraint>,
    pub name: Name,
    pub params: Vec<(Name, Type)>,
}

#[derive(Debug, PartialEq)]
pub struct EnumElem {
    pub name: Name,
    pub params: Vec<Type>,
}

#[derive(Debug, PartialEq)]
pub struct EnumData {
    pub generic_params: Vec<Constraint>,
    pub name: Name,
    pub elems: Vec<EnumElem>,
}

#[derive(Debug, PartialEq)]
pub struct Constraint {
    pub type_var: Type,
    pub traits: Vec<Type>,
}

// Types
pub type Type = WithPos<TypeNode>;

#[derive(Debug, PartialEq)]
pub enum TypeNode {
    Primary(Name), // int, bool, SomeStruct, SomeEnum
    Instantiate(Box<Type>, Vec<Type>), // List!(int), Option!(bool)
    FuncPtr(FuncPtrData), // func(int, bool): String
}

#[derive(Debug, PartialEq)]
pub struct InstantiateData {
    pub base: Box<Type>,
    pub args: Vec<Type>,
}

#[derive(Debug, PartialEq)]
pub struct FuncPtrData {
    pub params: Vec<Type>,
    pub ret: Box<Type>,
}

// module
#[derive(Debug)]
pub struct Module {
    pub name: Name,
    pub defs: Vec<Def>,
}
