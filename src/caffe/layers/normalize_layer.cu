#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void kForward(const int dim, const int count, const Dtype* divisor, 
		const Dtype* dividend, Dtype* res)
	{
		CUDA_KERNEL_LOOP(index, count){
			int n = index / dim;
			res[index] = dividend[index] / divisor[n];
		}
	}

	template <typename Dtype>
	__global__ void kBackward(const int dim, const int count, const Dtype* top_diff, const Dtype* norm,
		const Dtype* top_data, const Dtype* diff_data_product, Dtype* bottom_diff)
	{
		CUDA_KERNEL_LOOP(index, count){
			int n = index / dim;
			bottom_diff[index] = (top_diff[index] - top_data[index] * diff_data_product[n]) / norm[n];
		}
	}
template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  int num = bottom[0]->num();
  caffe_gpu_powx(num * dim_, bottom_data, Dtype(2), squared_data);
  caffe_gpu_gemv(CblasNoTrans, num, dim_, Dtype(1.0), squared_data, sum_multiplier_.gpu_data(),
	  Dtype(0.0), norm_data);
  caffe_gpu_powx(num, norm_.gpu_data(), Dtype(0.5), norm_data);
  kForward<Dtype> << <CAFFE_GET_BLOCKS(num * dim_), CAFFE_CUDA_NUM_THREADS >> >(dim_, num * dim_,
	  norm_.gpu_data(), bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
 /* for (int i=0; i<n; ++i) {
    caffe_gpu_asum<Dtype>(d, squared_data+i*d, &normsqr);
    caffe_gpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);
  }*/
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* norm_data = norm_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int num = top[0]->num();
  //Reuse the memory of squared_data
  Dtype* diff_data_product = squared_.mutable_gpu_data();
  caffe_gpu_mul(num * dim_, top_diff, top_data, diff_data_product);
  caffe_gpu_gemv(CblasNoTrans, num, dim_, Dtype(1.0), diff_data_product, sum_multiplier_.gpu_data(),
	  Dtype(0.0), diff_data_inner_product_.mutable_gpu_data());
  kBackward<Dtype> << <CAFFE_GET_BLOCKS(num * dim_), CAFFE_CUDA_NUM_THREADS >> >(dim_, num * dim_,
	  top_diff, norm_data, top_data, diff_data_inner_product_.gpu_data(), bottom_diff);
  CUDA_POST_KERNEL_CHECK;
 /* Dtype a;
  for (int i=0; i<n; ++i) {
    caffe_gpu_dot(d, top_data+i*d, top_diff+i*d, &a);
    caffe_gpu_scale(d, a, top_data+i*d, bottom_diff+i*d);
    caffe_gpu_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
    caffe_gpu_dot(d, bottom_data+i*d, bottom_data+i*d, &a);
    caffe_gpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
  }*/
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe
