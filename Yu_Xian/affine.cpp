#include "gdal_priv.h"
#include "gdal_alg.h"
#include <iostream>

int main() {
    GDALAllRegister();

    // Build GCP array (from your Python output)
    const int nGCPs = 4;
    GDAL_GCP gcps[nGCPs];
    GDALInitGCPs(nGCPs, gcps);

    gcps[0].dfGCPPixel = 286.3194698544767;
    gcps[0].dfGCPLine  = 179.90326587910673;
    gcps[0].dfGCPX     = -36.305995794973285;
    gcps[0].dfGCPY     = 20.272646683617168;
    gcps[0].dfGCPZ     = 0.0;

    gcps[1].dfGCPPixel = 433.5231564370652;
    gcps[1].dfGCPLine  = 429.4386233816361;
    gcps[1].dfGCPX     = -107.5442642659645;
    gcps[1].dfGCPY     = 0.9027544300838741;
    gcps[1].dfGCPZ     = 0.0;

    gcps[2].dfGCPPixel = 297.57930968062493;
    gcps[2].dfGCPLine  = 19.038790946231934;
    gcps[2].dfGCPX     = -117.87832528760572;
    gcps[2].dfGCPY     = -26.574107491124813;
    gcps[2].dfGCPZ     = 0.0;

    gcps[3].dfGCPPixel = 734.5881376368823;
    gcps[3].dfGCPLine  = 24.92611128273711;
    gcps[3].dfGCPX     = -16.840484988951744;
    gcps[3].dfGCPY     = -56.03853248692091;
    gcps[3].dfGCPZ     = 0.0;

    // Create refine transformer (order=1 polynomial, tolerance=40, min=3 GCPs)
    void *hTransform = GDALCreateGCPRefineTransformer(
        nGCPs, gcps, 1, FALSE, 40.0, 3
    );

    if (!hTransform) {
        std::cerr << "Failed to create GCP refine transformer" << std::endl;
        return 1;
    }

    // 3️⃣ Predict georeferenced coordinates at each GCP
    for (int i = 0; i < nGCPs; i++) {
        double x, y, z;
        int success;  // <-- this must be int
        int ret = GDALGCPTransform(
            hTransform, FALSE,
            1,                // number of points
            &gcps[i].dfGCPPixel,
            &gcps[i].dfGCPLine,
            &z,
            &success          // <-- pass int*
        );
    
        if (ret) {
            std::cout << "GCP " << i
                      << " Predicted = (" << x << ", " << y << ")"
                      << " | Actual = (" << gcps[i].dfGCPX << ", " << gcps[i].dfGCPY << ")"
                      << std::endl;
        } else {
            std::cout << "GCP " << i << " transform failed" << std::endl;
        }
    }
    

    // 4️⃣ Clean up
    GDALDestroyGCPTransformer(hTransform);
    GDALDeinitGCPs(nGCPs, gcps);

    return 0;
}