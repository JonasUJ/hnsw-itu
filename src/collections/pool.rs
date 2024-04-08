use std::sync::Mutex;

use crate::Pool;
use crate::Reset;

#[derive(Debug)]
pub struct VisitedPool<T>
where
    T: Reset,
{
    items: Vec<T>,
    pool_guard: Mutex<u8>,
}

impl<T> VisitedPool<T>
where
    T: Reset + Clone,
{
    pub fn new(size: usize, generator: impl Fn() -> T) -> Self {
        Self {
            items: vec![generator(); size],
            pool_guard: Mutex::new(0),
        }
    }
}

impl<T> Pool<T> for VisitedPool<T>
where
    T: Reset,
{
    fn get(&mut self, generator: impl Fn() -> T) -> T {
        let _l = self.pool_guard.lock();

        if let Some(mut t) = self.items.pop() {
            t.reset();
            t
        } else {
            generator()
        }
    }

    fn release(&mut self, t: T) {
        let _l = self.pool_guard.lock();

        self.items.push(t)
    }
}
