W jet is clustered using FastJet

It goes through selection process 
  1. Angular displacement: clustered jet (eta,phi) should be within 0.5 apart from W (eta,phi)
  2. Jet pT has to be within the pT bin that one's looking
  3. If more jets are identified, compare angular distances among jets and use the closest to W

Then selected jet goes through reclustering using C/A algo

Reclustering identify subjet signature which is used for jet image making

Identified W jets goes through preprocessing
  1. Translation
  2. Rotation
  3. Reflection

This code does not include pixelization(separated process) and zooming(only if pT bin is narrow, unnecessary step)
