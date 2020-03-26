
#include <assert.h>

#include <chrono>
#include <numeric>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "example_utils.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

static memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

static void depthwise_one(
    const engine &eng, stream &s, const std::vector<float> input_value,
    const std::vector<float> kernel_value, std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args, int zconv_in_hw,
    int zconv_out_hw, int stride, int pad) {
    const memory::dim batch = 1;
    // conv
    const int zconvz_i = 1, zconvz_o = 1;
    memory::dims zconvz_input_tz = {batch, zconvz_i, zconv_in_hw, zconv_in_hw};
    memory::dims zconvz_weights_tz = {zconvz_o, zconvz_i, 3, 3};
    memory::dims zconvz_bias_tz = {zconvz_o};
    memory::dims zconvz_dst_tz = {batch, zconvz_o, zconv_out_hw, zconv_out_hw};
    memory::dims zconvz_strides = {stride, stride};
    memory::dims zconvz_padding = {pad, pad};
    std::vector<float> zconvz_input(input_value);
    std::vector<float> zconvz_weights(kernel_value);
    std::vector<float> zconvz_bias(product(zconvz_bias_tz));
    auto zconvz_input_memory =
        memory({{zconvz_input_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(zconvz_input.data(), zconvz_input_memory);
    auto zconvz_user_weights_memory =
        memory({{zconvz_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(zconvz_weights.data(), zconvz_user_weights_memory);
    auto zconvz_bias_memory = memory({{zconvz_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(zconvz_bias.data(), zconvz_bias_memory);
    auto zconvz_src_md = memory::desc({zconvz_input_tz}, dt::f32, tag::nchw);
    auto zconvz_bias_md = memory::desc({zconvz_bias_tz}, dt::f32, tag::x);
    auto zconvz_weights_md =
        memory::desc({zconvz_weights_tz}, dt::f32, tag::oihw);
    auto zconvz_dst_md = memory::desc({zconvz_dst_tz}, dt::f32, tag::nchw);
    auto zconvz_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct,
        zconvz_src_md, zconvz_weights_md, zconvz_bias_md, zconvz_dst_md,
        zconvz_strides, zconvz_padding, zconvz_padding);
    auto zconvz_prim_desc =
        convolution_forward::primitive_desc(zconvz_desc, eng);
    auto zconvz_src_memory = zconvz_input_memory;
    if (zconvz_prim_desc.src_desc() != zconvz_input_memory.get_desc()) {
        zconvz_src_memory = memory(zconvz_prim_desc.src_desc(), eng);
        net.push_back(reorder(zconvz_input_memory, zconvz_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, zconvz_input_memory},
                            {DNNL_ARG_TO, zconvz_src_memory}});
    }
    auto zconvz_weights_memory = zconvz_user_weights_memory;
    if (zconvz_prim_desc.weights_desc() !=
        zconvz_user_weights_memory.get_desc()) {
        zconvz_weights_memory = memory(zconvz_prim_desc.weights_desc(), eng);
        reorder(zconvz_user_weights_memory, zconvz_weights_memory)
            .execute(s, zconvz_user_weights_memory, zconvz_weights_memory);
    }
    auto zconvz_dst_memory = memory(zconvz_prim_desc.dst_desc(), eng);
    net.push_back(convolution_forward(zconvz_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, zconvz_src_memory},
                        {DNNL_ARG_WEIGHTS, zconvz_weights_memory},
                        {DNNL_ARG_BIAS, zconvz_bias_memory},
                        {DNNL_ARG_DST, zconvz_dst_memory}});
}

static void simple_net(engine::kind engine_kind) {
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    const int zconv_in_hw = 4, zconv_out_hw = 4, stride = 1, pad = 1;
    std::vector<std::vector<float>> input_value{
        {0.5069131118594469, 0.17923216946722498, 0.34537740301767494,
         0.471699580702026, 0.5001322554993409, 0.9081130877372173,
         0.5627819983083859, 0.40658295290151836, 0.3220239005980854,
         0.15134760125459568, 0.25791994919254857, 0.08271506494115943,
         0.019071604301690193, 0.25087202354815785, 0.9568074053878342,
         0.40379526170010926},
        {0.29967796381176903, 0.8700719465133795, 0.31889930680889,
         0.7427104517732799, 0.2296249429467323, 0.04369669702465939,
         0.4716418286386287, 0.276197525104953, 0.30664729411196623,
         0.04166034015657549, 0.6455874088101299, 0.9150689288158794,
         0.40742423963241925, 0.18569437903573593, 0.13566888756056672,
         0.53553961700595},
        {0.16634674396673443, 0.4598003547356828, 0.2844935777340535,
         0.28509476300587067, 0.42485120051889824, 0.9543619465955848,
         0.8723125921671621, 0.29913831752188746, 0.6270707289242332,
         0.5062827082726463, 0.8230590555681602, 0.8066543861571697,
         0.38851523411704325, 0.9207844923904763, 0.4374602455441492,
         0.08879166004294936}};
    std::vector<std::vector<float>> kernel_value{
        {-0.1380029320716858, -0.06439334154129028, 0.18992996215820312,
         -0.0676359236240387, 0.3317890763282776, 0.19894033670425415,
         -0.20506946742534637, -0.21175572276115417, -0.15897111594676971},
        {-0.20507043600082397, -0.14640724658966064, -0.36556899547576904,
         -0.08639100193977356, -0.07023268938064575, 0.060543984174728394,
         -0.14337462186813354, 0.015974074602127075, 0.13187509775161743},
        {0.32964760065078735, 0.32043641805648804, -0.2134435921907425,
         0.29972171783447266, 0.1201443076133728, 0.011217653751373291,
         0.13423901796340942, -0.20793497562408447, 0.13924872875213623}};
    assert(input_value.size() == kernel_value.size() && "mismatch");
    for (size_t i = 0; i < input_value.size(); i++) {
        depthwise_one(eng, s, input_value.at(i), kernel_value.at(i), net,
                      net_args, zconv_in_hw, zconv_out_hw, stride, pad);
    }
    //[Create network]

    //[Execute model]
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
    }

    std::vector<float> conv_desc_vec(zconv_out_hw * zconv_out_hw);
    for (size_t i = 0; i < net.size(); ++i) {
        read_from_dnnl_memory(conv_desc_vec.data(),
                              net_args.at(i)[DNNL_ARG_DST]);
        printf("\tdst value: ");
        for (auto x : conv_desc_vec) {
            printf("%f, ", x);
        }
        printf("\n");
    }
    //[Execute model]
    s.wait();
}

int main() {
    try {
        simple_net(parse_engine_kind(1, NULL));
		printf("Intel(R) DNNL: depthwise: passed\n");
    } catch (error &e) {
        printf("Intel(R) DNNL: depthwise: failed!!!\n");
    }
    return 0;
}
