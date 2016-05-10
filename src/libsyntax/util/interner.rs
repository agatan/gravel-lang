use ast::Name;

use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct RcStr {
    string: Rc<String>,
}

impl RcStr {
    pub fn new(string: &str) -> RcStr {
        RcStr { string: Rc::new(string.to_string()) }
    }
}

impl fmt::Display for RcStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.string.fmt(f)
    }
}

impl Borrow<str> for RcStr {
    fn borrow(&self) -> &str {
        &self.string[..]
    }
}

impl Deref for RcStr {
    type Target = str;
    fn deref(&self) -> &str {
        &self.string[..]
    }
}

/// A "StrInterner" is a data structure that associates values with usize tags
/// and allows bidirectional lookup;
pub struct StrInterner {
    map: RefCell<HashMap<RcStr, Name>>,
    vect: RefCell<Vec<RcStr>>,
}

impl StrInterner {
    pub fn new() -> StrInterner {
        StrInterner {
            map: RefCell::new(HashMap::new()),
            vect: RefCell::new(Vec::new()),
        }
    }

    pub fn prefill(init: &[&str]) -> StrInterner {
        let rv = StrInterner::new();
        for &v in init {
            rv.intern(v);
        }
        rv
    }

    pub fn intern(&self, val: &str) -> Name {
        let mut map = self.map.borrow_mut();
        match map.get(val) {
            Some(&idx) => return idx,
            None => {}
        }

        let new_idx = Name(self.len() as u32);
        let val = RcStr::new(val);
        map.insert(val.clone(), new_idx);
        self.vect.borrow_mut().push(val);
        new_idx
    }

    /// Generate a new symbol.
    /// Generated symbol is anonymous, so can not be used for name lookup.
    pub fn gensym(&self, val: &str) -> Name {
        let new_idx = Name(self.len() as u32);
        self.vect.borrow_mut().push(RcStr::new(val));
        new_idx
    }

    pub fn get(&self, idx: Name) -> RcStr {
        self.vect.borrow()[idx.0 as usize].clone()
    }

    pub fn len(&self) -> usize {
        self.vect.borrow().len()
    }

    pub fn find<Q: ?Sized>(&self, val: &Q) -> Option<Name>
        where RcStr: Borrow<Q>,
              Q: Eq + Hash
    {
        match self.map.borrow().get(val) {
            Some(v) => Some(*v),
            None => None,
        }
    }

    pub fn clear(&self) {
        *self.map.borrow_mut() = HashMap::new();
        *self.vect.borrow_mut() = Vec::new();
    }

    pub fn reset(&self, other: StrInterner) {
        *self.map.borrow_mut() = other.map.into_inner();
        *self.vect.borrow_mut() = other.vect.into_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Name;

    #[test]
    #[should_panic]
    fn empty() {
        let i = StrInterner::new();
        i.get(Name(3));
    }

    #[test]
    fn intern_test() {
        let i = StrInterner::new();
        assert_eq!(i.intern("dog"), Name(0));
        assert_eq!(i.intern("dog"), Name(0));
        assert_eq!(i.intern("cat"), Name(1));
        assert_eq!(i.gensym("dog"), Name(2));
        assert_eq!(i.gensym("bird"), Name(3));

        assert_eq!(i.get(Name(0)), RcStr::new("dog"));
        assert_eq!(i.get(Name(1)), RcStr::new("cat"));
    }
}
