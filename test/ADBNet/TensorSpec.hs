{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module ADBNet.TensorSpec
    ( spec
    ) where

import           ADBNet.Tensor
import           Data.Array.Base
import           Test.Hspec
import           Test.Hspec.QuickCheck
import           Test.QuickCheck

matmulTest = matmul m v `shouldBe` e  where
    m :: Matrix Int
    m = mnew (7, 5) [0 ..]
    v = vnew 5 [0 ..]
    e = vnew 5 [ sum $ zipWith (*) [i .. i + 4] [0 .. 4] | i <- [0, 5 ..] ]

outerpTest = outerp u v `shouldBe` e  where
    u :: Vector Int
    u = tnew 7 [0 ..]
    v = tnew 5 [7 ..]
    e = tnew (7, 5) [ i * j | i <- [0 ..], j <- [7 .. (7 + 4)] ]

transpTest = transp m `shouldBe` e
  where
    m :: Matrix Int
    m = mnew (7, 5) [ i * j | i <- [0, 2 ..], j <- [0 .. 4] ]
    e = mnew (5, 7) [ i * j | i <- [0 ..], j <- [0, 2 .. 12] ]

fromRowsTest = m `shouldBe` e
  where
    m :: Matrix Int
    m = fromRows [ vnew 5 [i ..] | i <- [0, 5 .. 6 * 5] ]
    e = mnew (7, 5) [0 ..]

fromColsTest = m `shouldBe` e
  where
    m :: Matrix Int
    m = fromCols [ vnew 5 [i ..] | i <- [0, 5 .. 6 * 5] ]
    e = mnew (5, 7) [ i + j | i <- [0 ..], j <- [0, 5 .. 6 * 5] ]

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
