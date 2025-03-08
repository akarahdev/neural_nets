use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use serde::{
    Deserialize, Serialize,
    de::{Error, SeqAccess, Visitor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Array<T, const N: usize> {
    inner: [T; N],
}

impl<T: Default, const N: usize> Default for Array<T, N> {
    fn default() -> Self {
        Self {
            inner: std::array::from_fn(|_| T::default()),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Array<T, N> {
    fn from(value: [T; N]) -> Self {
        Array { inner: value }
    }
}

impl<T, const N: usize> From<Array<T, N>> for [T; N] {
    fn from(value: Array<T, N>) -> Self {
        value.inner
    }
}

impl<T, const N: usize> Deref for Array<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, const N: usize> DerefMut for Array<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: Serialize, const N: usize> Serialize for Array<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, T: Default + Deserialize<'de>, const N: usize> Deserialize<'de> for Array<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor {
            _phantom: PhantomData,
        })
    }
}

pub struct ArrayVisitor<'a, 'de, T: Default + Deserialize<'de>, const N: usize> {
    _phantom: PhantomData<(&'a (), &'de (), T)>,
}

impl<'de, T: Default + Deserialize<'de>, const N: usize> Visitor<'de>
    for ArrayVisitor<'_, 'de, T, N>
{
    type Value = Array<T, N>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("array of length ")?;
        formatter.write_str(&N.to_string())
    }

    #[inline]
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut arr = Array {
            inner: std::array::from_fn(|_| T::default()),
        };
        for idx in 0..N {
            match seq.next_element().unwrap() {
                Some(val) => arr[idx] = val,
                None => return Err(Error::invalid_length(idx, &self)),
            }
        }
        Ok(arr)
    }
}
