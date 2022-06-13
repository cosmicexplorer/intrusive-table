/*
 * Description: A hash (?) table that allows allocating memory within a specified memory region.
 *
 * Copyright (C) 2022 Danny McClanahan <dmcC2@hypnicjerk.ai>
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! A hash (?) table that allows allocating memory within a specified memory region.

#![no_std]
/* NB: THIS IS A no_std CRATE!!!!! */
#![deny(warnings)]

use core::hash::{Hash, Hasher};
use core::mem;
use core::slice;
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum Error {
  NoMoreSpace(usize),
  OutOfHashableSpots(usize),
  DeleteDidNotExist,
}

pub trait AllocationDescriptor: Default + Eq + PartialEq + Hash {
  fn digest(slice: &[u8]) -> Self;
  fn size_of_pointed_to(&self) -> usize;
}

pub trait IntrusiveTable<K> {
  fn initialize(&mut self);
  fn retrieve(&self, key: &K) -> Option<&[u8]>;
  fn allocate(&mut self, source: &[u8]) -> Result<(&K, &[u8]), Error>;
  fn delete(&mut self, key: &K) -> Result<(), Error>;
}

#[derive(Default)]
struct TableEntry<K: AllocationDescriptor> {
  pub key: K,
  pub offset: AtomicUsize,
}

impl<K: AllocationDescriptor> TableEntry<K> {
  pub fn is_default(&self) -> bool {
    self.key.size_of_pointed_to() == 0
  }
}

pub struct IntrusiveAllocator<'a, K: AllocationDescriptor> {
  hash_table: &'a mut [TableEntry<K>],
  allocated_region_extent: &'a AtomicUsize,
  allocatable_data: &'a mut [u8],
}

struct Offset(usize);

pub const HASH_TABLE_SPACE_FACTOR: usize = 10;

pub fn hash_usize<T: Hash>(t: &T) -> usize {
  /* FIXME: SipHasher is deprecated, but there appears to be no alternative Hasher implementation in
   * core? */
  #[allow(deprecated)]
  let mut hasher = core::hash::SipHasher::new();
  Hash::hash(t, &mut hasher);

  hasher.finish() as usize
}

impl<'a, K: 'a + AllocationDescriptor> IntrusiveAllocator<'a, K> {
  ///
  /// The entry point for creating an intrusive allocator.
  ///
  pub fn allocator_within_region(owned_region: &'a mut [u8]) -> Self {
    Self::get_layout(owned_region)
  }

  ///
  /// The table is laid out in three consecutive segments as:
  ///
  /// ^[hash table][<extent index>][allocatable data]$
  ///
  fn get_layout(owned_region: &'a mut [u8]) -> Self {
    let hash_table_space = Self::hash_table_space(owned_region);
    let hash_table_num_entries = Self::hash_table_num_entries(owned_region);

    /* [hash table][[extent][allocatable]] */
    let (hash_table_data, extent_and_allocatable_data) =
      owned_region.split_at_mut(hash_table_space);
    let hash_table = unsafe {
      let table_ptr = mem::transmute::<*mut u8, *mut TableEntry<K>>(hash_table_data.as_mut_ptr());
      slice::from_raw_parts_mut(table_ptr, hash_table_num_entries)
    };

    /* [extent][allocatable] */
    let (extent_data, allocatable_data) =
      extent_and_allocatable_data.split_at_mut(mem::size_of::<AtomicUsize>());

    let allocated_region_extent: &'a AtomicUsize = unsafe {
      let extent_ptr = mem::transmute::<*const u8, *const AtomicUsize>(extent_data.as_ptr());
      &*extent_ptr
    };

    IntrusiveAllocator {
      hash_table,
      allocated_region_extent,
      allocatable_data,
    }
  }

  /* These methods describe aspects of the intrusive region that can be computed without reading
   * the data inside it yet. */
  fn full_region_size(owned_region: &mut [u8]) -> usize {
    owned_region.len()
  }

  fn hash_table_space(owned_region: &mut [u8]) -> usize {
    Self::full_region_size(owned_region) / HASH_TABLE_SPACE_FACTOR
  }

  fn hash_table_num_entries(owned_region: &mut [u8]) -> usize {
    Self::hash_table_space(owned_region) / mem::size_of::<TableEntry<K>>()
  }

  /* These methods actually query/traverse the table. */
  fn table_num_entries(&self) -> usize {
    self.hash_table.len()
  }

  fn hash_key(&self, key: &K) -> usize {
    /* Ensure the result points to an entry within the table. */
    hash_usize(&key) % self.table_num_entries()
  }

  fn atomic_get_offset(entry: &TableEntry<K>) -> Offset {
    Offset(entry.offset.load(Ordering::SeqCst))
  }

  fn get_offset_slice(&self, offset: Offset, key: &K) -> &[u8] {
    let Offset(begin) = offset;
    let end = begin + key.size_of_pointed_to();
    &self.allocatable_data[begin..end]
  }

  fn atomic_get_slice(&self, entry: &TableEntry<K>) -> &[u8] {
    let offset = Self::atomic_get_offset(entry);
    self.get_offset_slice(offset, &entry.key)
  }

  fn find_existing_entry(&self, key: &K) -> Option<&TableEntry<K>> {
    /* Incredibly basic linear probing. */
    /* FIXME: probe in a ring past the end of the array! */
    for cur_entry in self.hash_table[self.hash_key(key)..].iter() {
      if cur_entry.is_default() {
        break;
      }
      if cur_entry.key == *key {
        return Some(cur_entry);
      }
    }
    return None;
  }

  fn find_entry_to_delete(&mut self, key: &K) -> Result<(), Error> {
    let initial_index = self.hash_key(key);
    /* Incredibly basic linear probing. */
    /* FIXME: probe in a ring past the end of the array! */
    for cur_entry in self.hash_table[initial_index..].iter_mut() {
      /* The entry did not exist in the table!! */
      if cur_entry.is_default() {
        break;
      }
      if cur_entry.key == *key {
        /* FIXME: does this need to be written atomically as well? */
        cur_entry.key = K::default();
        /* TODO: just use TableEntry::default()? */
        cur_entry.offset.store(0, Ordering::SeqCst);
        return Ok(());
      }
    }
    Err(Error::DeleteDidNotExist)
  }

  fn find_first_empty_or_matching_entry(&mut self, source: &[u8]) -> Result<(&K, &[u8]), Error> {
    let key = K::digest(source);
    let initial_index = self.hash_key(&key);
    let IntrusiveAllocator {
      hash_table,
      allocated_region_extent,
      allocatable_data,
    } = self;
    let table_len = hash_table.len();
    /* Incredibly basic linear probing. */
    /* FIXME: probe in a ring past the end of the array! */
    for cur_entry in hash_table[initial_index..].iter_mut() {
      /* The entry did *not* exist already -- let's populate it. */
      if cur_entry.is_default() {
        /* Atomically bump up the memory line. */
        let previous_extent =
          allocated_region_extent.fetch_add(key.size_of_pointed_to(), Ordering::SeqCst);
        let new_extent = previous_extent + key.size_of_pointed_to();
        /* If we run out of space, error out. */
        if new_extent > allocatable_data.len() {
          return Err(Error::NoMoreSpace(allocatable_data.len()));
        }
        let new_region = &mut allocatable_data[previous_extent..new_extent];
        /* Write the source data to the new region. */
        new_region.copy_from_slice(source);
        /* Update the entry so that it can be retrieved later. */
        cur_entry.offset.store(previous_extent, Ordering::SeqCst);
        /* FIXME: does this need to be written atomically as well? */
        cur_entry.key = key;
        return Ok((&cur_entry.key, new_region));
      }
      /* If the entry already existed, return the existing slice for it. */
      if cur_entry.key == key {
        let Offset(begin) = Self::atomic_get_offset(cur_entry);
        let end = begin + key.size_of_pointed_to();
        return Ok((&cur_entry.key, &allocatable_data[begin..end]));
      }
    }
    /* If the entry couldn't be allocated, error out. */
    Err(Error::OutOfHashableSpots(table_len))
  }
}

impl<'a, K: AllocationDescriptor> IntrusiveTable<K> for IntrusiveAllocator<'a, K> {
  fn initialize(&mut self) {
    /* Apparently this will compile down to vectorized operations -- see
     * https://stackoverflow.com/questions/51732596/what-is-the-equivalent-of-a-safe-memset-for-slices/51732799#51732799 */
    for entry in self.hash_table.iter_mut() {
      *entry = TableEntry::<K>::default();
    }
    self.allocated_region_extent.store(0, Ordering::SeqCst);
  }

  fn retrieve(&self, key: &K) -> Option<&[u8]> {
    self.find_existing_entry(key).map(|cur_entry| {
      /* We have definitely found an entry. This may be at the same time as another process, so we
       * load the atomic pointer. */
      self.atomic_get_slice(cur_entry)
    })
  }

  fn allocate(&mut self, source: &[u8]) -> Result<(&K, &[u8]), Error> {
    self.find_first_empty_or_matching_entry(source)
  }

  fn delete(&mut self, key: &K) -> Result<(), Error> {
    self.find_entry_to_delete(key)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
  struct Key {
    pub id: u32,
    pub size: usize,
  }
  impl AllocationDescriptor for Key {
    fn digest(slice: &[u8]) -> Self {
      let id = hash_usize(&slice) as u32;
      let size = slice.len();
      Key { id, size }
    }
    fn size_of_pointed_to(&self) -> usize {
      self.size
    }
  }

  fn get_backing_bytes() -> [u8; 500] {
    [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0,
    ]
  }

  #[test]
  fn allocate_retrieve_delete_end_to_end() -> Result<(), Error> {
    let mut backing_bytes: [u8; 500] = get_backing_bytes();
    let mut allocator = IntrusiveAllocator::<'_, Key>::allocator_within_region(&mut backing_bytes);
    allocator.initialize();

    let source_bytes = "asdfasdfasdf".as_bytes();
    let key = Key::digest(source_bytes);

    assert_eq!(None, allocator.retrieve(&key));
    assert_eq!(Err(Error::DeleteDidNotExist), allocator.delete(&key));

    let (ret_key, ret_bytes) = allocator.allocate(source_bytes)?;
    assert_eq!(*ret_key, key);
    assert_eq!(source_bytes, ret_bytes);
    assert_ne!(source_bytes.as_ptr(), ret_bytes.as_ptr());

    assert_eq!(source_bytes, allocator.retrieve(&key).unwrap());

    allocator.delete(&key)?;

    assert_eq!(None, allocator.retrieve(&key));
    assert_eq!(Err(Error::DeleteDidNotExist), allocator.delete(&key));
    Ok(())
  }

  #[test]
  fn allocate_too_large() -> Result<(), Error> {
    let mut backing_bytes: [u8; 500] = get_backing_bytes();
    let mut allocator = IntrusiveAllocator::<'_, Key>::allocator_within_region(&mut backing_bytes);
    allocator.initialize();

    let source_bytes = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".as_bytes();
    assert!(source_bytes.len() > 500);

    /* Returns the amount of space that was in total available to allocate. */
    assert_eq!(
      Err(Error::NoMoreSpace(442)),
      allocator.allocate(source_bytes)
    );
    Ok(())
  }

  #[test]
  fn allocate_too_many() -> Result<(), Error> {
    let mut backing_bytes: [u8; 500] = get_backing_bytes();
    let mut allocator = IntrusiveAllocator::<'_, Key>::allocator_within_region(&mut backing_bytes);
    allocator.initialize();

    let mut errored: bool = false;
    for i in 0..50 {
      /* Generate a non-repeating sequence of bytes. */
      let mut i = i as u32;
      let source_bytes: &mut [u8] =
        unsafe { slice::from_raw_parts_mut(mem::transmute::<&mut u32, *mut u8>(&mut i), 4) };
      match allocator.allocate(source_bytes) {
        Ok(_) => (),
        Err(Error::OutOfHashableSpots(x)) => {
          /* Assert that this failure occurs because there are only two spots available to allocate
           * in the hash table in total. */
          assert_eq!(x, 2);
          errored = true;
        }
        Err(_) => unreachable!(),
      };
    }
    assert!(errored);
    Ok(())
  }

  #[test]
  fn allocate_can_pick_up_where_left_off() -> Result<(), Error> {
    let mut backing_bytes: [u8; 500] = get_backing_bytes();

    let source_bytes = "asdfasdfasdf".as_bytes();
    let key = Key::digest(source_bytes);

    {
      let mut allocator =
        IntrusiveAllocator::<'_, Key>::allocator_within_region(&mut backing_bytes);
      allocator.initialize();

      assert_eq!(None, allocator.retrieve(&key));
      assert_eq!(Err(Error::DeleteDidNotExist), allocator.delete(&key));

      allocator.allocate(source_bytes)?;
    }

    {
      let allocator = IntrusiveAllocator::<'_, Key>::allocator_within_region(&mut backing_bytes);
      assert_eq!(source_bytes, allocator.retrieve(&key).unwrap());
    }

    Ok(())
  }
}
