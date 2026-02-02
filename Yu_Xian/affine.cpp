#include "gdal.h"
#include "gdal_alg.h"
#include "cpl_conv.h"  // for CPLMalloc/CPLFree
#include <cstdio>

int main()
{
    GDALAllRegister();

    // -----------------------------
    // 1. Build GCP array
    // -----------------------------
    const int nGCPs = 4;
    GDAL_GCP gcps[nGCPs];
    GDALInitGCPs(nGCPs, gcps);

    gcps[0].dfGCPPixel = 622.2811890466377;
    gcps[0].dfGCPLine  = 102.49433703793743;
    gcps[0].dfGCPX     = 149.1094311969085;
    gcps[0].dfGCPY     = 17.564646767351825;
    gcps[0].dfGCPZ     = 0.0;

    gcps[1].dfGCPPixel = 137.6266560291843;
    gcps[1].dfGCPLine  = 540.2528119295063;
    gcps[1].dfGCPX     = 141.90773272559804;
    gcps[1].dfGCPY     = 18.19144038171389;
    gcps[1].dfGCPZ     = 0.0;

    gcps[2].dfGCPPixel = 570.1446619661385;
    gcps[2].dfGCPLine  = 84.92714352482984;
    gcps[2].dfGCPX     = 145.57399902035402;
    gcps[2].dfGCPY     = 17.41234388683788;
    gcps[2].dfGCPZ     = 0.0;

    gcps[3].dfGCPPixel = 599.5177271454444;
    gcps[3].dfGCPLine  = 366.1259940916348;
    gcps[3].dfGCPX     = 146.12063681237203;
    gcps[3].dfGCPY     = 16.444029815693412;
    gcps[3].dfGCPZ     = 0.0;

    // -----------------------------
    // 2. Create GCP transformer
    // -----------------------------
    int nReqOrder   = 1;      // affine
    int bReversed   = FALSE;  // pixel/line -> geo
    double tolerance = 0;   // 0 = GDAL default (no pruning)
    int minGcps     = 3;

    void* hTransform = GDALCreateGCPTransformer(
        nGCPs,
        gcps,
        nReqOrder,
        bReversed
        // tolerance,
        // minGcps
    );

    if (hTransform == nullptr)
    {
        fprintf(stderr, "Failed to create GCP refine transformer\n");
        GDALDeinitGCPs(nGCPs, gcps);
        return 1;
    }

    // -----------------------------
    // 3. Apply the transformer
    // -----------------------------
    double x = 400.0;   // pixel
    double y = 200.0;   // line
    double z = 0.0;
    int success = TRUE;

    GDALGCPTransform(
        hTransform,
        FALSE,   // forward transform
        1,
        &x,
        &y,
        &z,
        &success
    );

    if (!success)
    {
        fprintf(stderr, "Transform failed\n");
    }
    else
    {   
        printf("Test point:\n");
        printf("Transformed point:\n");
        printf("X = %.10f\n", x);
        printf("Y = %.10f\n", y);
    }

    for (int j = 0; j < nGCPs; j += 1){

        double xp = gcps[j].dfGCPPixel;
        double yp = gcps[j].dfGCPLine;
        double zp = 0.0;

        GDALGCPTransform(
            hTransform,
            FALSE,
            1,
            &xp,
            &yp,
            &zp,
            &success
        );
        if (!success)
        {
        fprintf(stderr, "Transform failed\n");
        }
        else
        {
        printf("GCP point %d\n", j);
        printf("Transformed point:\n");
        printf("X = %.10f\n", xp);
        printf("Y = %.10f\n", yp);
        }
    }


    // -----------------------------
    // 4. Cleanup
    // -----------------------------
    GDALDestroyGCPTransformer(hTransform);
    GDALDeinitGCPs(nGCPs, gcps);

    return 0;
}