#![allow(dead_code)]
#![allow(unused)]
extern crate mpi;

use ndarray::{arr2, Array, Array2, ArrayView, Dim, s};

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as f32;
    let rank = world.rank();

    if rank == 0 {
        let p_sqrt = size.sqrt() as i32;

        let a = arr2(&[
            [2, 1, 5, 3, 4, 7],
            [0, 7, 1, 6, 8, 3],
            [9, 2, 4, 4, 1, 1],
            [3, 6, 7, 2, 2, 5],
            [3, 6, 7, 2, 2, 5],
            [3, 6, 7, 2, 2, 5],
        ]);

        let b = arr2(&[
            [6, 1, 2, 3, 8, 1],
            [4, 5, 6, 5, 2, 9],
            [1, 9, 8, -8, -3, 0],
            [4, 0, -8, 5, 3, 1],
            [4, 0, -8, 5, 3, 1],
            [4, 0, -8, 5, 3, 1],
        ]);
        println!("Hello from rank init");
        let a_slices = split(&a, p_sqrt);
        let b_slices = split(&b, p_sqrt);
        // println!("{:?}", a_slices);
        // println!("{:}?", a_slices[0].dot(&a_slices[1]))
        a_slices.iter().enumerate().for_each(|(idx, item)| {
            println!("{:?}", item.to_owned().as_slice().unwrap());
            world.process_at_rank(idx as i32).send(item.to_owned().as_slice().unwrap());
        })
    } else {
        let (mut msg, _status) = world.process_at_rank(0).receive_vec::<i32>();
        let new_arr = Array2::from_shape_vec((3, 3), msg.to_vec()).unwrap();
        println!("Hello from rank {:} {:?}", world.rank(), new_arr);
    }

    world.barrier();

    println!("Wololo")
    // let c = arr2(&[
    //     [33, 52, 26, -14],
    //     [53, 44, 2, 57],
    //     [82, 55, 30, 25],
    //     [57, 96, 82, -7],
    // ]);

    // let calculated_c = a.dot(&b);
    // println!("{}", calculated_c.slice(s![0..2,0..2]));
    // assert_eq!(c, calculated_c)
}

fn split(matrix: &Array2<i32>, parts: i32) -> Vec<ArrayView<i32, Dim<[usize; 2]>>> {
    let dim_x = matrix.dim().0 as i32;
    let dim_y = matrix.dim().1 as i32;
    let mut sliced = vec![];

    for i in 0..parts {
        for j in 0..parts {
            sliced.push(matrix.slice(
                s![i*(dim_x/parts)..i*(dim_x/parts)+(dim_x/parts),
                    j*(dim_y/parts)..j*(dim_y/parts)+(dim_y/parts)]));
        }
    }
    sliced
}