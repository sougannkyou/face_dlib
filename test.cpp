#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;


double time_now(){
    struct timeval time;
    if(gettimeofday(&time,NULL)){
        return  0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) try{
    cout << "************** test ********************\n" << endl;
//    char buffer[256];
//    getcwd(buffer,256);
//    cout << "pwds:" << buffer << endl;

    if (argc != 3){
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./test ../faces/bald_guys.jpg ../faces/jason.jpg" << endl;
        cout << endl;
        return 1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    // ---------------------------------------------------------------------------------------------
    matrix<rgb_pixel> img;
    load_image(img, argv[1]);

//    image_window win(img);

    double start = time_now();
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img)){
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // win.add_overlay(face);
    }
    if (faces.size() == 0){
        cout << "[test] Not detect any faces in test image!" << endl;
        return 1;
    }
    cout << "[test] detect " << faces.size() << " faces in test image. Total:" <<
    time_now() - start << "s, Avg:" <<(time_now() - start) / faces.size() << "s." << endl;
    // ---------------------------------------------------------------------------------------------
    start = time_now();
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);
    cout << "[test] Encoding Total: " <<
    time_now() - start  << "s. Avg:" <<(time_now() - start) / faces.size() << "s." << endl;
    // ---------------------------------------------------------------------------------------------
    start = time_now();
    matrix<rgb_pixel> img_target;
    load_image(img_target, argv[2]);
    std::vector<matrix<rgb_pixel>> faces_target;
    for (auto face_target : detector(img_target)){
        auto shape_target = sp(img_target, face_target);
        matrix<rgb_pixel> face_chip_target;
        extract_image_chip(img_target, get_face_chip_details(shape_target,150,0.25), face_chip_target);
        faces_target.push_back(face_chip_target);
        //win.add_overlay(face);
    }
    if (faces_target.size() == 0){
        cout << "[target] Not detect any faces in target image!" << endl;
        return 1;
    }
    cout << "[target] detect " << faces_target.size() << " faces in target. Total: " << time_now() - start << "s." << endl;
    // ---------------------------------------------------------------------------------------------
    start = time_now();
    std::vector<matrix<float,0,1>> face_descriptors_target = net(faces_target); // one target face
    cout << "[target] Encoding: " << time_now() - start << "s." << endl;
    // ---------------------------------------------------------------------------------------------
    start = time_now();
    std::vector<matrix<rgb_pixel>> temp;
    for (size_t i = 0; i < face_descriptors.size(); ++i){
        if (length(face_descriptors[i] - face_descriptors_target[0]) < 0.6){
            temp.push_back(faces[i]);
            cout << "[match] recognition test No." << i << endl;
        }
    }
    if(temp.size() > 0){
        cout << "[match] recognition " << temp.size() << " faces in target. Total: " <<
        time_now() - start << "s." << endl;
    }else{
        cout << "[match] not recognition any target faces." << endl;
    }
    // ---------------------------------------------------------------------------------------------
    std::vector<image_window> win_clusters(1);
    win_clusters[0].set_title("recognition faces");
    win_clusters[0].set_image(tile_images(temp));

    cout << "hit enter to terminate: "<< endl;
    cin.get();
}catch (std::exception& e){
    cout << e.what() << endl;
}


