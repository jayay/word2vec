#[cfg(feature = "simd")]
extern crate packed_simd;

#[cfg(feature = "simd")]
use self::packed_simd::f32x4;

#[cfg(feature = "simd")]
pub fn dot_product(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len() % 4, 0);

    x.chunks_exact(4)
        .map(f32x4::from_slice_unaligned)
        .zip(y.chunks_exact(4).map(f32x4::from_slice_unaligned))
        .map(|(x, y)| x * y)
        .sum::<f32x4>()
        .sum()
}

#[cfg(not(feature = "simd"))]
pub fn dot_product(arr1: &[f32], arr2: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    for (elem1, elem2) in arr1.iter().zip(arr2.iter()) {
        result += elem1 * elem2;
    }
    result
}

#[cfg(feature = "simd")]
pub fn vector_norm(vector: &mut Vec<f32>) {
    assert!(vector.len() % 4 == 0);

    let sum = 1.0
        / vector
            .chunks_exact(4)
            .map(f32x4::from_slice_unaligned)
            .map(|x| x * x)
            .sum::<f32x4>()
            .sum()
            .sqrt();

    for mut chunk in vector.chunks_exact_mut(4) {
        let mut simd = f32x4::from_slice_unaligned(&mut chunk);
        simd *= sum;
        unsafe {
            simd.write_to_slice_unaligned_unchecked(&mut chunk)
        }
    }
}

#[cfg(not(feature = "simd"))]
pub fn vector_norm(vector: &mut [f32]) {
    let sum = 1.0 / vector.iter().fold(0f32, |sum, &x| sum + (x * x)).sqrt();
    for x in vector.iter_mut() {
        (*x) *= sum;
    }
}

/// Get the mean (average) of the given Iterator of numbers
pub fn mean<Iterable: Iterator<Item = f32>>(numbers: Iterable) -> f32 {
    let (sum, count) = numbers.fold((0f32, 0), |(sum, count), x| (sum + x, count + 1));
    sum / (count as f32)
}


#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_dot_product() {
        let x = vec![0.12345, 0.56789, 0.4, -434.4];
        let y = vec![0.2453, -0.234234, -0.9323, 0.1235];
        assert!((-54.124058 - dot_product(&x, &y)).abs() < 0.000005);
    }

     #[bench]
    fn bench_dot_product(b: &mut Bencher) {
        let x = test::black_box(vec![
0.068_479_03,
-0.003_140_191_5,
-0.019_410_182,
0.008_659_908,
-0.047_384_944,
-0.091_880_98,
0.044_363_964,
-0.008_436_143,
-0.221_836,
0.079_327_17,
0.052_283_91,
0.051_507_924,
0.026_866_235,
0.042_886_227,
-0.150_447_95,
0.080_151_37,
-0.168_410_24,
0.040_421_274,
-0.048_473_61,
0.032_074_787,
0.107_494_034,
0.096_662_08,
-0.056_023_66,
0.127_867_22,
0.152_719_89,
-0.220_813_45,
0.316_548_88,
-0.094_375_93,
0.042_061_016,
-0.039_828_613,
0.013_013_678,
-0.057_035_62,
0.170_578_9,
0.107_264_884,
-0.069_435_66,
0.056_060_113,
0.075_160_47,
0.000_722_760_33,
-0.035_978_958,
0.095_466_73,
0.017_190_387,
0.211_598_01,
0.114_840_47,
-0.013_076_136,
0.085_607_335,
0.039_410_252,
0.000_248_339_46,
0.094_898_69,
-0.030_429_995,
-0.097_357_66,
0.050_982_26,
0.242_490_23,
0.044_477_385,
0.101_763_74,
-0.022_047_602,
-0.045_035_77,
0.035_170_734,
-0.104_933_485,
0.033_919_446,
0.061_091_196,
0.056_548_856,
-0.125_138_36,
0.095_899_954,
0.164_212_87,
-0.062_717_44,
-0.024_114_914,
-0.016_013_792,
0.190_090_46,
0.176_511_82,
0.043_404_695,
0.035_255_738,
0.036_759_555,
-0.118_789_524,
-0.053_981_666,
0.053_118_546,
-0.079_510_98,
0.003_044_293_3,
-0.018_724_19,
-0.068_301_95,
0.026_909_612,
0.095_571_44,
0.085_695_42,
-0.064_969_35,
0.096_377_335,
0.043_329_16,
0.053_797_368,
0.032_627_98,
-0.018_402_599,
0.147_991_97,
0.068_842_29,
0.005_214_586_4,
0.163_774_54,
0.050_389_25,
-0.073_637_48,
-0.272_154_63,
0.037_323_263,
0.136_039_12,
0.137_882_96,
0.020_492_628,
-0.093_078_19]);
        let y = test::black_box(vec![0.084_598_24,
-0.031_929_04,
-0.035_995_677,
0.019_117_568,
-0.113_580_77,
-0.021_020_407,
-0.013_204_093,
0.095_338_486,
-0.201_278_95,
0.050_777_238,
0.014_905_112,
-0.005_027_082,
0.080_857_866,
-0.037_310_876,
-0.065_772_325,
0.105_478_406,
-0.127_983_17,
0.035_149_984,
0.050_002_87,
0.028_460_834,
0.063_969_076,
-0.053_924_132,
-0.056_944_06,
0.092_618_86,
0.114_757_14,
-0.170_280_64,
0.234_777_38,
-0.040_971_432,
0.077_273_75,
-0.129_965_17,
0.058_084_242,
-0.063_790_84,
0.195_613_37,
0.168_713_78,
-0.084_341_526,
-0.006_961_72,
-0.025_382_465,
0.026_607_556,
0.019_785_73,
0.114_247_605,
-0.039_363_38,
0.192_852_29,
0.087_393_396,
-0.059_394_7,
-0.037_323_046,
0.016_437_778,
0.024_722_412,
0.182_384_36,
-0.051_981_095,
-0.124_323_01,
0.025_443_753,
0.324_617_45,
0.041_502_547,
0.066_627_1,
-0.009_171_247,
-0.011_023_498,
0.098_790_2,
-0.078_202_16,
-0.001_535_856_6,
0.040_706_012,
0.090_905_4,
-0.120_286_24,
0.103_047_92,
0.263_289,
0.071_300_84,
-0.007_235_344_5,
-0.005_536_441_7,
0.125_683_86,
0.111_744_04,
0.009_066_494,
0.012_060_294,
-0.034_856_804,
-0.044_523_597,
-0.009_079_151,
0.000_890_543_7,
-0.123_038_59,
0.005_558_813_5,
0.035_552_785,
-0.142_161_03,
-0.035_736_08,
0.089_289_2,
0.112_836_055,
-0.142_529_26,
0.069_331_11,
0.050_782_673,
0.004_590_401_4,
0.017_084_349,
-0.010_786_293,
0.057_857_99,
0.101_426_795,
0.025_846_712,
0.192_916_23,
0.017_345_587,
0.002_314_033_5,
-0.320_702_64,
0.027_905_198,
0.131_941_26,
0.072_478_615,
0.002_479_001_2,
-0.071_388_74]);
        b.iter(|| dot_product(&x, &y));
    }

    #[bench]
    fn bench_vector_norm(b: &mut Bencher) {
        let mut v = test::black_box(vec![0.084_598_24,
-0.031_929_04,
-0.035_995_677,
0.019_117_568,
-0.113_580_77,
-0.021_020_407,
-0.013_204_093,
0.095_338_486,
-0.201_278_95,
0.050_777_238,
0.014_905_112,
-0.005_027_082,
0.080_857_866,
-0.037_310_876,
-0.065_772_325,
0.105_478_406,
-0.127_983_17,
0.035_149_984,
0.050_002_87,
0.028_460_834,
0.063_969_076,
-0.053_924_132,
-0.056_944_06,
0.092_618_86,
0.114_757_14,
-0.170_280_64,
0.234_777_38,
-0.040_971_432,
0.077_273_75,
-0.129_965_17,
0.058_084_242,
-0.063_790_84,
0.195_613_37,
0.168_713_78,
-0.084_341_526,
-0.006_961_72,
-0.025_382_465,
0.026_607_556,
0.019_785_73,
0.114_247_605,
-0.039_363_38,
0.192_852_29,
0.087_393_396,
-0.059_394_7,
-0.037_323_046,
0.016_437_778,
0.024_722_412,
0.182_384_36,
-0.051_981_095,
-0.124_323_01,
0.025_443_753,
0.324_617_45,
0.041_502_547,
0.066_627_1,
-0.009_171_247,
-0.011_023_498,
0.098_790_2,
-0.078_202_16,
-0.001_535_856_6,
0.040_706_012,
0.090_905_4,
-0.120_286_24,
0.103_047_92,
0.263_289,
0.071_300_84,
-0.007_235_344_5,
-0.005_536_441_7,
0.125_683_86,
0.111_744_04,
0.009_066_494,
0.012_060_294,
-0.034_856_804,
-0.044_523_597,
-0.009_079_151,
0.000_890_543_7,
-0.123_038_59,
0.005_558_813_5,
0.035_552_785,
-0.142_161_03,
-0.035_736_08,
0.089_289_2,
0.112_836_055,
-0.142_529_26,
0.069_331_11,
0.050_782_673,
0.004_590_401_4,
0.017_084_349,
-0.010_786_293,
0.057_857_99,
0.101_426_795,
0.025_846_712,
0.192_916_23,
0.017_345_587,
0.002_314_033_5,
-0.320_702_64,
0.027_905_198,
0.131_941_26,
0.072_478_615,
0.002_479_001_2,
-0.071_388_74]);
        b.iter(|| vector_norm(&mut v));
    }
}
