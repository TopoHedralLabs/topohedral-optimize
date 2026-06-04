//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

const UNSET: f64 = f64::MIN;

pub struct CircularBuffer
{
    values: Vec<f64>,
    read_ptr: usize,
    write_ptr: usize,
    len: usize,
}

impl CircularBuffer
{
    pub fn new(num_elems: usize) -> Self
    {
        assert!(num_elems > 0, "circular buffer capacity must be non-zero");

        Self {
            values: vec![UNSET; num_elems],
            read_ptr: 0,
            write_ptr: 0,
            len: 0,
        }
    }

    pub fn append(
        &mut self,
        new_value: f64,
    )
    {
        self.values[self.write_ptr] = new_value;
        self.write_ptr = self.next(self.write_ptr);

        if self.len == self.capacity()
        {
            self.read_ptr = self.next(self.read_ptr);
        }
        else
        {
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize
    {
        self.len
    }

    pub fn capacity(&self) -> usize
    {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool
    {
        self.len == 0
    }

    pub fn is_full(&self) -> bool
    {
        self.len == self.capacity()
    }

    pub fn get(
        &self,
        idx: usize,
    ) -> Option<f64>
    {
        if idx >= self.len
        {
            return None;
        }

        Some(self.values[(self.read_ptr + idx) % self.capacity()])
    }

    pub fn newest(&self) -> Option<f64>
    {
        if self.is_empty()
        {
            return None;
        }

        let idx = (self.write_ptr + self.capacity() - 1) % self.capacity();
        Some(self.values[idx])
    }

    pub fn max(&self) -> Option<f64>
    {
        self.iter().reduce(f64::max)
    }

    pub fn oldest(&self) -> Option<f64>
    {
        self.get(0)
    }

    pub fn next(
        &self,
        ptr: usize,
    ) -> usize
    {
        (ptr + 1) % self.capacity()
    }

    pub fn iter(&self) -> CircularBufferIter<'_>
    {
        CircularBufferIter {
            buffer: self,
            idx: 0,
        }
    }
}

pub struct CircularBufferIter<'a>
{
    buffer: &'a CircularBuffer,
    idx: usize,
}

impl Iterator for CircularBufferIter<'_>
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item>
    {
        let value = self.buffer.get(self.idx)?;
        self.idx += 1;

        Some(value)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn new_buffer_is_empty_with_fixed_capacity()
    {
        let buffer = CircularBuffer::new(3);

        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 3);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.oldest(), None);
        assert_eq!(buffer.newest(), None);
        assert_eq!(buffer.get(0), None);
        assert_eq!(buffer.iter().collect::<Vec<_>>(), Vec::<f64>::new());
    }

    #[test]
    fn appends_values_until_full_without_changing_order()
    {
        let mut buffer = CircularBuffer::new(3);

        buffer.append(1.0);
        buffer.append(2.0);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_full());
        assert_eq!(buffer.oldest(), Some(1.0));
        assert_eq!(buffer.newest(), Some(2.0));
        assert_eq!(buffer.get(0), Some(1.0));
        assert_eq!(buffer.get(1), Some(2.0));
        assert_eq!(buffer.get(2), None);
        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![1.0, 2.0]);

        buffer.append(3.0);

        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn overwrites_oldest_value_after_capacity_is_reached()
    {
        let mut buffer = CircularBuffer::new(3);

        for value in [1.0, 2.0, 3.0, 4.0, 5.0]
        {
            buffer.append(value);
        }

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.capacity(), 3);
        assert!(buffer.is_full());
        assert_eq!(buffer.oldest(), Some(3.0));
        assert_eq!(buffer.newest(), Some(5.0));
        assert_eq!(buffer.get(0), Some(3.0));
        assert_eq!(buffer.get(1), Some(4.0));
        assert_eq!(buffer.get(2), Some(5.0));
        assert_eq!(buffer.get(3), None);
        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn max_returns_none_for_empty_buffer()
    {
        let buffer = CircularBuffer::new(3);

        assert_eq!(buffer.max(), None);
    }

    #[test]
    fn max_returns_largest_stored_value()
    {
        let mut buffer = CircularBuffer::new(4);

        buffer.append(-1.0);
        buffer.append(6.0);
        buffer.append(2.5);

        assert_eq!(buffer.max(), Some(6.0));
    }

    #[test]
    fn max_ignores_values_that_have_been_overwritten()
    {
        let mut buffer = CircularBuffer::new(3);

        for value in [100.0, 1.0, 2.0, 3.0]
        {
            buffer.append(value);
        }

        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.max(), Some(3.0));
    }

    #[test]
    fn single_element_buffer_always_keeps_latest_value()
    {
        let mut buffer = CircularBuffer::new(1);

        buffer.append(1.0);
        buffer.append(2.0);
        buffer.append(3.0);

        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.capacity(), 1);
        assert!(buffer.is_full());
        assert_eq!(buffer.oldest(), Some(3.0));
        assert_eq!(buffer.newest(), Some(3.0));
        assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![3.0]);
    }

    #[test]
    fn append_does_not_reallocate_storage()
    {
        let mut buffer = CircularBuffer::new(2);
        let values_ptr = buffer.values.as_ptr();

        for value in [1.0, 2.0, 3.0, 4.0]
        {
            buffer.append(value);
            assert_eq!(buffer.values.as_ptr(), values_ptr);
            assert_eq!(buffer.capacity(), 2);
        }
    }

    #[test]
    #[should_panic(expected = "circular buffer capacity must be non-zero")]
    fn zero_capacity_buffer_panics()
    {
        CircularBuffer::new(0);
    }
}
