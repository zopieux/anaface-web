use cached::proc_macro::cached;
use ndarray::{s, Axis, Dim, Dimension};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::js_sys::Float32Array;

type Array1 = ndarray::Array1<f32>;
type Array2 = ndarray::Array2<f32>;
type ArrayView2<'a> = ndarray::ArrayView2<'a, f32>;

fn array_from_js(arr: &Float32Array, r: usize, c: usize) -> Array2 {
    let mut o = Array1::zeros((r * c,));
    unsafe { arr.raw_copy_to_ptr(o.as_mut_ptr()) };
    o.to_shape(((r, c), ndarray::Order::RowMajor))
        .unwrap()
        .to_owned()
}

#[derive(tsify::Tsify, Serialize)]
#[tsify(into_wasm_abi)]
pub struct DetectedFace {
    pub detection_score: f32,
    pub bbox: [f32; 4],
    pub kpss: Vec<f32>,
}
#[derive(tsify::Tsify, Serialize)]
#[tsify(into_wasm_abi)]
pub struct DetectedFaces(Vec<DetectedFace>);

#[wasm_bindgen]
pub fn retinaface(
    s8: &Float32Array,
    s16: &Float32Array,
    s32: &Float32Array,
    b8: &Float32Array,
    b16: &Float32Array,
    b32: &Float32Array,
    k8: &Float32Array,
    k16: &Float32Array,
    k32: &Float32Array,
    size: usize,
    det_scale: f32,
    detection_threshold: f32,
    nms_threshold: f32,
) -> DetectedFaces {
    use itertools::MultiUnzip;
    use std::ops::Div;
    let s8 = array_from_js(s8, 12800, 1);
    let s16 = array_from_js(s16, 3200, 1);
    let s32 = array_from_js(s32, 800, 1);
    let b8 = array_from_js(b8, 12800, 4);
    let b16 = array_from_js(b16, 3200, 4);
    let b32 = array_from_js(b32, 800, 4);
    let k8 = array_from_js(k8, 12800, 10);
    let k16 = array_from_js(k16, 3200, 10);
    let k32 = array_from_js(k32, 800, 10);
    const N_ANCHORS: u32 = 2;
    let (all_score, all_bbox, all_kpss): (Vec<_>, Vec<_>, Vec<_>) = itertools::multizip((
        [8, 16, 32],
        &[s8, s16, s32],
        &[b8, b16, b32],
        &[k8, k16, k32],
    ))
    .flat_map(|(stride, scores, bbox, kpss)| -> Option<_> {
        let width = size as u32 / stride;
        let height = size as u32 / stride;
        let bbox = bbox * (stride as f32);
        let kpss = kpss * (stride as f32);
        let where_above: Vec<usize> = index_where(&scores.view(), |v| v >= detection_threshold)
            .map(|i| i.0)
            .collect();
        let score = scores
            .select(Axis(0), &where_above)
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let ac = anchor_centers(width, height, stride, N_ANCHORS);
        let bbox = distance_to_bbox(ac.view(), bbox.view());
        let bbox = bbox.select(Axis(0), &where_above);
        let kpss = distance_to_kpss(ac.view(), kpss.view());
        let kpss = kpss
            .to_shape((kpss.shape()[0], kpss.shape()[1] / 2, 2))
            .unwrap();
        let kpss = kpss.select(Axis(0), &where_above);
        Some((score, bbox, kpss))
    })
    .multiunzip();
    let score = ndarray::concatenate(Axis(0), &as_view(&all_score)).unwrap();
    let boxes = ndarray::concatenate(Axis(0), &as_view(&all_bbox))
        .unwrap()
        .div(det_scale);
    let kpss = ndarray::concatenate(Axis(0), &as_view(&all_kpss))
        .unwrap()
        .div(det_scale);

    let scores_ravel = score.to_shape((score.len(),)).unwrap();
    let order = argmax(scores_ravel.view());
    let order = order.as_slice().unwrap();

    let pre_det = ndarray::concatenate![Axis(1), boxes, score];
    let pre_det = pre_det.select(Axis(0), order);
    let keep = nms(pre_det.view(), nms_threshold);
    let det = pre_det.select(Axis(0), &keep);
    let kpss = kpss.select(Axis(0), order).select(Axis(0), &keep);
    DetectedFaces(
        det.axis_iter(Axis(0))
            .zip(kpss.axis_iter(Axis(0)))
            .map(|(d, kpss)| DetectedFace {
                detection_score: d[4],
                bbox: [d[0], d[1], d[2], d[3]],
                kpss: kpss.flatten().to_vec(),
            })
            .collect(),
    )
}

#[cfg(any())]
#[wasm_bindgen]
pub fn get_bgr_pixels_3_height_width(
    data: &Uint8Array,
    width: usize,
    height: usize,
) -> Float32Array {
    use ndarray::Array3;

    const MEAN: f32 = 127.5;
    const STD: f32 = 128.0;

    // Reshape the input data into a 3D array with shape (height, width, 4)
    let mut o = ndarray::Array1::<u8>::zeros((height * width * 4,));
    unsafe { data.raw_copy_to_ptr(o.as_mut_ptr()) };
    let rgba_array: ndarray::Array3<u8> = o
        .to_shape(((width, height, 4), ndarray::Order::RowMajor))
        .unwrap()
        .to_owned();

    // Remove the alpha channel by slicing, resulting in shape (height, width, 3)
    let rgb_array = rgba_array.slice(s![.., .., 0..3]);

    // Swap dimensions to get shape (3, height, width)
    let mut bgr_array = Array3::<u8>::zeros((3, height, width));
    bgr_array
        .slice_mut(s![0, .., ..])
        .assign(&rgb_array.slice(s![.., .., 2])); // B channel
    bgr_array
        .slice_mut(s![1, .., ..])
        .assign(&rgb_array.slice(s![.., .., 1])); // G channel
    bgr_array
        .slice_mut(s![2, .., ..])
        .assign(&rgb_array.slice(s![.., .., 0])); // R channel

    // Normalize the pixel values using broadcasting
    let bgr_f32 = bgr_array.mapv(|x| (x as f32 - MEAN) / STD);
    let out = Float32Array::new_with_length((3 * width * height) as u32);
    out.copy_from(bgr_f32.as_slice_memory_order().unwrap());
    out
}

#[cached]
fn anchor_centers(width: u32, height: u32, step: u32, repeat: u32) -> Array2 {
    Array2::from_shape_fn(((width * height * repeat) as usize, 2), |(i, j)| {
        let i = i as u32;
        ((if j == 0 {
            (i / repeat) % width
        } else {
            (i / repeat) / width
        }) * step) as f32
    })
}

fn as_view<'a, A, D>(
    arr: &'a [ndarray::ArrayBase<ndarray::OwnedRepr<A>, D>],
) -> Vec<ndarray::ArrayBase<ndarray::ViewRepr<&'a A>, D>>
where
    D: ndarray::Dimension + 'a,
{
    arr.iter().map(|e| e.view()).collect()
}

fn index_where<'a, F, T, D>(
    arr: &'a ndarray::ArrayView<T, D>,
    pred: F,
) -> std::iter::FilterMap<
    ndarray::iter::IndexedIter<'a, T, D>,
    impl FnMut((<D as Dimension>::Pattern, &'a T)) -> Option<<D as Dimension>::Pattern>,
>
where
    D: ndarray::Dimension,
    F: Fn(T) -> bool,
    T: Copy,
{
    arr.indexed_iter()
        .filter_map(move |(i, v)| pred(*v).then_some(i))
}

fn argmax<A>(arr: ndarray::ArrayView<'_, A, Dim<[usize; 1]>>) -> ndarray::Array1<usize>
where
    A: PartialOrd,
{
    use itertools::Itertools;
    arr.indexed_iter()
        .sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap())
        .map(|(i, _)| i)
        .collect()
}

fn distance_to_bbox(points: ArrayView2, distance: ArrayView2) -> Array2 {
    use std::ops::{Add, Sub};
    let x1 = points.slice(s![.., 0]).sub(&distance.slice(s![.., 0]));
    let y1 = points.slice(s![.., 1]).sub(&distance.slice(s![.., 1]));
    let x2 = points.slice(s![.., 0]).add(&distance.slice(s![.., 2]));
    let y2 = points.slice(s![.., 1]).add(&distance.slice(s![.., 3]));
    ndarray::stack![Axis(1), x1, y1, x2, y2]
}

fn distance_to_kpss(points: ArrayView2, distance: ArrayView2) -> Array2 {
    use std::ops::Add;
    let end = distance.shape()[1];
    let mut out = vec![];
    for i in (0..end).step_by(2) {
        let px = points.slice(s![.., i % 2]).add(&distance.slice(s![.., i]));
        let py = points
            .slice(s![.., i % 2 + 1])
            .add(&distance.slice(s![.., i + 1]));
        out.push(px);
        out.push(py);
    }
    let what: Vec<_> = out.iter().map(|e| e.view()).collect();
    ndarray::stack(Axis(1), &what).unwrap()
}

fn nms(dets: ArrayView2, iou_thres: f32) -> Vec<usize> {
    let n = dets.nrows();
    if n == 0 {
        return Vec::new();
    }

    // Create vector of indices sorted by score in descending order
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&i, &j| {
        dets.get((j, 4))
            .unwrap()
            .partial_cmp(dets.get((i, 4)).unwrap())
            .unwrap()
    });

    let mut keep = Vec::new();

    while !order.is_empty() {
        // Keep the box with highest score
        let i = order[0];
        keep.push(i);

        // Remove boxes with IoU > threshold
        order = order[1..]
            .iter()
            .filter(|&&j| calculate_iou(dets, i, j) <= iou_thres)
            .copied()
            .collect();
    }

    keep
}

fn calculate_iou(dets: ArrayView2, i: usize, j: usize) -> f32 {
    let x1i = dets.get((i, 0)).unwrap();
    let y1i = dets.get((i, 1)).unwrap();
    let x2i = dets.get((i, 2)).unwrap();
    let y2i = dets.get((i, 3)).unwrap();

    let x1j = dets.get((j, 0)).unwrap();
    let y1j = dets.get((j, 1)).unwrap();
    let x2j = dets.get((j, 2)).unwrap();
    let y2j = dets.get((j, 3)).unwrap();

    // Calculate intersection coordinates
    let x1_inter = x1i.max(*x1j);
    let y1_inter = y1i.max(*y1j);
    let x2_inter = x2i.min(*x2j);
    let y2_inter = y2i.min(*y2j);

    // Calculate areas
    let width_i = x2i - x1i;
    let height_i = y2i - y1i;
    let area_i = width_i * height_i;

    let width_j = x2j - x1j;
    let height_j = y2j - y1j;
    let area_j = width_j * height_j;

    // Check if boxes overlap
    if x2_inter < x1_inter || y2_inter < y1_inter {
        return 0.0;
    }

    // Calculate intersection area
    let width_inter = x2_inter - x1_inter;
    let height_inter = y2_inter - y1_inter;
    let area_inter = width_inter * height_inter;

    // Calculate union area and return IoU
    let area_union = area_i + area_j - area_inter;
    area_inter / area_union
}
