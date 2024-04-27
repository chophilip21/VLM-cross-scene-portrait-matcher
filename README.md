# Interiormagic

The idea is to visualize your new furniture at home. This is the domain of 2D image synthesis, like virtual try on, guided by masks and another image target. But you can go one step further with 3D reconstruction after image synthesis. 

1. User takes a single photo of his/her room (let's call this source), and a photo of a furniture (let's call this reference) that it wants to potentially purchase.

2. Using models like SAM, user places a point on the source image to identify an object that he wants to replace, or simply a point where he wants to place the new furniture.  

3. Run cheap extraction on the furniture. This should be easy, because furnitures are often on plain background, and therefore we do not need complicated extraction model. 

4. Run guided image synthesis on the insertion point with the extracted target.

5. On the image synthesis result, run zero shot 3d reconstruction model for realistic experience. 