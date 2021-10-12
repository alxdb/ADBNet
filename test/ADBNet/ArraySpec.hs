module ADBNet.ArraySpec
    ( spec
    ) where

import           ADBNet.Array
import           Data.Array
import           Test.Hspec

azipTest1 = a `shouldBe` e
  where
    x = listArray (1, 5) [0 ..]
    y = listArray (2, 7) [0 ..]
    a = azip (+) x y
    e = listArray (1, 5) $ map (* 2) [0 ..]

azipTest2 = a `shouldBe` e
  where
    x = listArray (1, 5) [0 ..]
    y = listArray (4, 7) [0 ..]
    a = azip (+) x y
    e = listArray (4, 7) $ map (* 2) [0 ..]

spec = do
    describe "azip" $ do
        it "works for same sized arrays"        azipTest1
        it "works for differently sized arrays" azipTest2
