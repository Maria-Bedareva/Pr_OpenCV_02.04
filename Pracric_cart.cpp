#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void processCards(vector<Mat>& cardsImages, vector<string>& cardsNames, vector<Mat>& cardsDescriptors, vector<vector<KeyPoint>>& cardsKeypoints, Mat& image, Ptr<ORB> detector) {
    if (cardsImages.size() != cardsNames.size() || cardsImages.size() != cardsDescriptors.size()) {
        cerr << "Error with size" << endl;
        return;
    }

    Ptr<BFMatcher> matcher = BFMatcher::create();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Mat gray, blur, canny;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(5, 5), 0);
    Canny(blur, canny, 120, 255);

    findContours(canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 1000) 
            continue;

        vector<Point> approx;
        approxPolyDP(contour, approx, 0.02 * arcLength(contour, true), true);

        if (approx.size() != 4)
            continue;

        double ratio = fabs(1.0 - (approx[1].y - approx[0].y) / (approx[2].x - approx[1].x));
        if (ratio > 0.2) 
            continue;

        RotatedRect cardRect = minAreaRect(approx);

        Mat card;
        string cardName;
        Mat cardDescriptors;
        vector<KeyPoint> cardKeypoints;

        Mat rotatedMatrix, rotatedImage;
        rotatedMatrix = getRotationMatrix2D(cardRect.center, cardRect.angle, 1.0);
        warpAffine(image, rotatedImage, rotatedMatrix, image.size(), INTER_CUBIC);
        getRectSubPix(rotatedImage, cardRect.size, cardRect.center, card);

        detector->detectAndCompute(card, noArray(), cardKeypoints, cardDescriptors);

        if (cardDescriptors.empty()) {
            cardName = "";
        }
        else {
            int maxI = -1;
            int maxCount = 0;

            for (int i = 0; i < cardsImages.size(); i++) {
                if (cardsDescriptors[i].empty()) {
                    continue;
                }

                vector<vector<DMatch>> knn_matches;
                matcher->knnMatch(cardsDescriptors[i], cardDescriptors, knn_matches, 3);

                vector<DMatch> correct;

                for (size_t i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                        correct.push_back(knn_matches[i][0]);
                    }
                }

                if (maxCount < correct.size()) {
                    maxCount = static_cast<int>(correct.size());
                    maxI = i;
                }
            }

            if (maxI == -1) {
                cardName = "";
            }
            else {
                cardName = cardsNames[maxI];
            }
        }

        if (cardName != "") {
            drawContours(image, vector<vector<Point>>{approx}, -1, Scalar(0, 0, 255), 2);
            putText(image, cardName, cardRect.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        }
    }
}

int main() {
    vector<Mat> cardsImages;
    vector<string> cardsNames = {
        "9_cherv", "6_kresti", "6_pik", "7_bubi", "dama_cherv", "dama_pik", "tuz_kresti", "tuz_bubi", "korol_pik", "valet_bubi"
    };
    vector<Mat> cardsDescriptors;
    vector<vector<KeyPoint>> cardsKeypoints;

    Ptr<ORB> detector = ORB::create();

    Mat card;
    for (const auto& cardName : cardsNames) {
        card = imread("C:/Users/User/Desktop/Карты/" + cardName + ".jpg");
        if (!card.empty()) {
            resize(card, card, Size(150, 220));
            cardsImages.push_back(card);
            Mat descriptors;
            vector<KeyPoint> keypoints;
            detector->detectAndCompute(card, noArray(), keypoints, descriptors);
            cardsDescriptors.push_back(descriptors);
            cardsKeypoints.push_back(keypoints);
        }
    }

    Mat image = imread("C:/Users/User/Desktop/Карты/cart.jpg");

    processCards(cardsImages, cardsNames, cardsDescriptors, cardsKeypoints, image, detector);

    imshow("Final image", image);
    waitKey(0);

    return 0;
}
