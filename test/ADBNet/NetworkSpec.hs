module ADBNet.NetworkSpec
    ( spec
    ) where

import           ADBNet.Misc
import           ADBNet.Network
import           ADBNet.Tensor
import           Data.Array
import           Debug.Trace
import           System.Random
import           Test.Hspec

activationsTest = a `shouldBe` e
  where
    (nn, g) = randomNetwork [2, 4, 1] (mkStdGen 0)
    i       = fst $ trnd 2 (0.0, 1.0) g
    a       = activations nn i
    z1      = matmul (nn ! 1) (vapp i 1.0)
    a1      = tmap sigmoid z1
    z2      = matmul (nn ! 2) (vapp a1 1.0)
    a2      = tmap sigmoid z2
    e       = (listArray (1, 2) [a1, a2], listArray (1, 2) [z1, z2])

deltasTest = a `shouldBe` e
  where
    (nn, s0) = randomNetwork [2, 4, 1] (mkStdGen 0)
    (i , s1) = trnd 2 (0.0, 1.0) s0
    o        = fst $ trnd 1 (0.0, 1.0) s1
    (as, zs) = activations nn i
    d2       = tmap sigmoid' (zs ! 2) * tzip rss' (as ! 2) o
    d1       = tmap sigmoid' (zs ! 1) * transp (remCol (nn ! 2)) `matmul` d2
    e        = listArray (1, 2) [d1, d2]
    a        = deltas nn (as, zs) o

gradientsTest = a `shouldBe` e
  where
    (nn, s0) = randomNetwork [2, 4, 1] (mkStdGen 0)
    (i , s1) = trnd 2 (0.0, 1.0) s0
    o        = fst $ trnd 1 (0.0, 1.0) s1
    (as, zs) = activations nn i
    ds       = deltas nn (as, zs) o
    a        = gradients nn i (as, zs) ds
    g1       = (ds ! 1) `outerp` vapp i 1.0
    g2       = (ds ! 2) `outerp` vapp (as ! 1) 1.0
    e        = listArray (1, 2) [g1, g2]

gradientsDimsTest = map dims (elems a) `shouldBe` map dims (elems nn)
  where
    (nn, s0) = randomNetwork [2, 4, 1] (mkStdGen 0)
    (i , s1) = trnd 2 (0.0, 1.0) s0
    o        = fst $ trnd 1 (0.0, 1.0) s1
    (as, zs) = activations nn i
    ds       = deltas nn (as, zs) o
    a        = gradients nn i (as, zs) ds

singleFitTest = a `shouldBe` e
  where
    (nn, s0)   = randomNetwork [2, 4, 1] (mkStdGen 0)
    (i , s1)   = trnd 2 (0.0, 1.0) s0
    (o , _ )   = trnd 1 (0.0, 1.0) s1
    (as, zs)   = activations nn i
    ds         = deltas nn (as, zs) o
    gs         = gradients nn i (as, zs) ds
    a = array (1, 2) [ (i, (nn ! i) - scale 0.5 (gs ! i)) | i <- [1 .. 2] ]
    (e, l, o') = singleFit 0.5 nn (i, o)

trainTest = zipWith loss a e `shouldSatisfy` all (< 0.1)
  where
    nn' = randomNetwork_ [2, 4, 1] 0
    td =
        [ (tnew 2 [0.0, 0.0], tnew 1 [0.0])
        , (tnew 2 [1.0, 0.0], tnew 1 [1.0])
        , (tnew 2 [0.0, 1.0], tnew 1 [1.0])
        , (tnew 2 [1.0, 1.0], tnew 1 [0.0])
        ]
    tp = TrainingParameters { learningRate = 0.2, trainingIterations = 1600 }
    (nn, _) = train tp nn' td
    a       = traceShowId $ map (run nn . fst) td
    e       = traceShowId $ map snd td

spec = do
    describe "activations" $ do
        it "acts as expected" activationsTest
    describe "deltas" $ do
        it "acts as expected" deltasTest
    describe "gradients" $ do
        it "acts as expected"                            gradientsTest
        it "produces result with the correct dimensions" gradientsDimsTest
    describe "singleFit" $ do
        it "acts as expected" singleFitTest
    describe "train" $ do
        it "can emulate xor" trainTest
