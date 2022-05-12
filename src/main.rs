#![allow(dead_code)]
#![allow(unused)]
extern crate mpi;

use ndarray::{arr2, s, Array, Array2, ArrayView, Dim};

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as f32;
    let rank = world.rank();
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
        [1, 9, 8, 8, 3, 0],
        [1, 1, 1, 1, 1, 1],
        [3, 2, 0, 6, 0, 1],
        [9, 1, 4, 1, 0, 1],
    ]);
    let a_slices = split(&a, p_sqrt);
    let b_slices = split(&b, p_sqrt);

    /// calculates next up and left ranks, based on knowledge,
    /// that processors are arranged in NxN mesh.
    let rank = world.rank();
    let mut left = 0;
    let mut up = 0;
    let mut right = 0;
    let mut down = 0;

    if rank % p_sqrt == 0 {
        left = rank + p_sqrt - 1;
    } else {
        left = rank - 1;
    }
    if (rank - p_sqrt + 1) % p_sqrt == 0 {
        right = rank - p_sqrt + 1;
    } else {
        right = rank + 1;
    }

    if rank - p_sqrt < 0 {
        up = world.size() - p_sqrt + rank;
    } else {
        up = rank - p_sqrt;
    }
    if rank + p_sqrt >= world.size() {
        down = -world.size() + p_sqrt + rank;
    } else {
        down = rank + p_sqrt;
    }
    let item_a = a_slices[rank as usize];
    let item_b = b_slices[rank as usize];
    // let mut result_c = Array::zeros((1, 1));
    for i in 0..p_sqrt {
        //Calculate byproduct

        //send A left
        println!("Rank {:} sending A to rank {:}", rank, left);
        world.process_at_rank(left as i32).send(item_a.to_owned().as_slice().unwrap());

        //send B up
        println!("Rank {:} sending B to rank {:}", rank, up);
        world.process_at_rank(up as i32).send(item_b.to_owned().as_slice().unwrap());


        //Receive A values
        println!("Rank {:} receiving A from rank {:}", rank, right);
        let (mut msg_a, _) = world.process_at_rank(right).receive_vec::<i32>();
        println!("{:?}", msg_a);

        //Receive B values
        println!("Rank {:} receiving B from rank {:}", rank, down);
        let (mut msg_b, _) = world.process_at_rank(down).receive_vec::<i32>();
        println!("{:?}", msg_b);

        //Reconstruct ndarrays
        let new_a = Array2::from_shape_vec(
            (a_slices[0].dim().0 as usize,
             a_slices[0].dim().1 as usize),
            msg_a.to_vec()).unwrap();

        let new_b = Array2::from_shape_vec(
            (b_slices[0].dim().0 as usize,
             b_slices[0].dim().1 as usize),
            msg_b.to_vec()).unwrap();

    }

    // println!("Rank {} got slice of a: {:?}", world.rank(), a_slices[world.rank() as usize]);

    // if rank == 0 {
    //     println!("Hello from rank init");
    //
    //
    //     a_slices.iter().enumerate().for_each(|(idx, item)| {
    //         println!("{:?}", item.to_owned().as_slice().unwrap());
    //         world.process_at_rank(idx as i32).send(item.to_owned().as_slice().unwrap());
    //     })
    // } else {
    //     let (mut msg, _status) = world.process_at_rank(0).receive_vec::<i32>();
    //     let new_arr = Array2::from_shape_vec(
    //         (dim.0/p_sqrt,dim.1/p_sqrt), // as usize
    //         msg.to_vec()).unwrap();
    //     println!("Hello from rank {:} {:?}", world.rank(), new_arr);
    // }

    world.barrier();
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
            sliced.push(matrix.slice(s![
                i * (dim_x / parts)..i * (dim_x / parts) + (dim_x / parts),
                j * (dim_y / parts)..j * (dim_y / parts) + (dim_y / parts)
            ]));
        }
    }
    sliced
}
