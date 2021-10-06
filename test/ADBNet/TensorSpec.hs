{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module ADBNet.TensorSpec
    ( spec
    ) where

import           ADBNet.Tensor
import           Test.Hspec

matmulTest = matmul m v `shouldBe` e  where
    m :: Matrix Int
    m = tnew (7, 5) [0 ..]
    v = tnew 5 [0 ..]
    e = tnew 7 [ sum $ zipWith (*) [i .. i + 4] [0 .. 4] | i <- [0, 5 ..] ]

outerpTest = outerp u v `shouldBe` e  where
    u :: Vector Int
    u = tnew 7 [0 ..]
    v = tnew 5 [7 ..]
    e = tnew (7, 5) [ i * j | i <- [0 ..], j <- [7 .. (7 + 4)] ]

transpTest = transp m `shouldBe` e
  where
    m :: Matrix Int
    m = tnew (7, 5) [ i * j | i <- [0, 2 ..], j <- [0 .. 4] ]
    e = tnew (5, 7) [ i * j | i <- [0 ..], j <- [0, 2 .. 12] ]

fromRowsTest = fromRows rs `shouldBe` e
  where
    rs :: [Vector Int]
    rs = [ tnew 5 [i ..] | i <- [0, 5 .. 6 * 5] ]
    e  = tnew (7, 5) [0 ..]

fromColsTest = fromCols cs `shouldBe` e
  where
    cs :: [Vector Int]
    cs = [ tnew 5 [i ..] | i <- [0, 5 .. 6 * 5] ]
    e  = tnew (5, 7) [ i + j | i <- [0 ..], j <- [0, 5 .. 6 * 5] ]

addColTest = addCol m c `shouldBe` e
  where
    m :: Matrix Int
    m = tnew (5, 7) [0 ..]
    c = tnew 5 [(5 * 7) ..]
    e = tnew
        (5, 8)
        [ if mod i 8 == 7 then rem i 7 + (5 * 7) else i - div i 8
        | i <- [0 ..]
        ]

remColTest = remCol m `shouldBe` e
  where
    m :: Matrix Int
    m = tnew (5, 7) [0 ..]
    e = tnew (5, 6) [ i + div i 6 | i <- [0 ..] ]

vappTest = vapp v x `shouldBe` e
  where
    v :: Vector Int
    v = tnew 5 [0 ..]
    x = 5
    e = tnew 6 [0 ..]

spec = do
    describe "matmul" $ do
        it "acts as expected" matmulTest
    describe "outerp" $ do
        it "acts as expected" outerpTest
    describe "transp" $ do
        it "acts as expected" transpTest
    describe "fromRows" $ do
        it "acts as expected" fromRowsTest
    describe "fromCols" $ do
        it "acts as expected" fromColsTest
    describe "addCol" $ do
        it "acts as expected" addColTest
    describe "remCol" $ do
        it "acts as expected" remColTest
    describe "vapp" $ do
        it "acts as expected" vappTest
