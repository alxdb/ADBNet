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

instance (Arbitrary e, IArray UArray e) => Arbitrary (Matrix e) where
    arbitrary = do
        d <- sized $ \n -> choose $ ixs (n, n)
        v <- vector (size d)
        return $ tnew d v

instance (Arbitrary e, IArray UArray e) => Arbitrary (Vector e) where
    arbitrary = do
        d <- sized $ \n -> choose $ ixs n
        v <- vector (size d)
        return $ tnew d v

matmulTest = matmul m v `shouldBe` e  where
    m :: Matrix Int
    m = mnew (7, 5) $ [ j | i <- [0 .. 6], j <- [0 .. 4] ]
    v = vnew 5 $ repeat 1
    e = vnew 5 [ sum (replicate 7 i) | i <- [0 .. 4] ]

outerpTest = outerp u v `shouldBe` e  where
    u :: Vector Int
    u = tnew 7 [0, 2 ..]
    v = tnew 5 [0 ..]
    e = tnew (7, 5) [ i * j | i <- [0, 2 ..], j <- [0 .. 4] ]

fromRowsTest = m `shouldBe` e
  where
    m :: Matrix Int
    m = fromRows [ vnew 5 [i ..] | i <- [0 .. 6] ]
    e = mnew (7, 5) [ i + j | i <- [0 .. 6], j <- [0 .. 4] ]

fromColsTest = m `shouldBe` e
  where
    m :: Matrix Int
    m = fromCols [ vnew 5 [i ..] | i <- [0 .. 6] ]
    e = mnew (5, 7) [ i + j | i <- [0 .. 4], j <- [0 .. 6] ]

spec = do
    describe "matmul" $ do
        it "acts as expected" matmulTest
    describe "outerp" $ do
        it "acts as expected" outerpTest
    describe "fromRows" $ do
        it "acts as expected" fromRowsTest
