module ADBNet.NetworkSpec
    ( spec
    ) where

import           ADBNet.Misc
import           ADBNet.Network
import           ADBNet.Tensor
import           Data.Array
import           System.Random
import           Test.Hspec

activationsTest = a `shouldBe` e
  where
    (nn, g) = randomNetwork [2, 4, 2] (mkStdGen 0)
    i       = fst $ trnd 2 (0.0, 1.0) g
    a       = activations nn i
    z1      = matmul (nn ! 1) (vapp i 1.0)
    a1      = tmap sigmoid z1
    z2      = matmul (nn ! 2) (vapp a1 1.0)
    a2      = tmap sigmoid z2
    e       = (listArray (1, 2) [a1, a2], listArray (1, 2) [z1, z2])

deltasTest = a `shouldBe` e
  where
    (nn, s0) = randomNetwork [2, 4, 2] (mkStdGen 0)
    (i , s1) = trnd 2 (0.0, 1.0) s0
    o        = fst $ trnd 2 (0.0, 1.0) s1
    (as, zs) = activations nn i
    d2       = tmap sigmoid' (zs ! 2) * tzip rss' (as ! 2) o
    d1       = tmap sigmoid' (zs ! 1) * transp (remCol (nn ! 2)) `matmul` d2
    e        = listArray (1, 2) [d1, d2]
    a        = deltas nn (as, zs) o

gradientsTest = a `shouldBe` e
  where
    (nn, s0) = randomNetwork [2, 4, 2] (mkStdGen 0)
    (i , s1) = trnd 2 (0.0, 1.0) s0
    o        = fst $ trnd 2 (0.0, 1.0) s1
    (as, zs) = activations nn i
    ds       = deltas nn (as, zs) o
    a        = gradients nn i (as, zs) ds
    g1       = (ds ! 1) `outerp` vapp i 1.0
    g2       = (ds ! 2) `outerp` vapp (as ! 1) 1.0
    e        = listArray (1, 2) [g1, g2]

spec = do
    describe "activations" $ do
        it "acts as expected" activationsTest
    describe "deltas" $ do
        it "acts as expected" deltasTest
    describe "gradients" $ do
        it "acts as expected" gradientsTest
