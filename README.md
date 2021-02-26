# Clouds

# DATASET DETAILS:
---
This dataset was procured from [Harvard Data Review] (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD).
---
## Dataset Contents:
- Cirrus        (139 images)
- Cirrostratus  (287 images)
- Cirrocumulus  (268 images)
- Altocumulus   (221 images)
- Altostratus   (188 images)
- Cumulus       (182 images)
- Cumulonimbus  (242 images)
- Nimbostratus  (274 images)
- Stratocumulus (340 images)
- Stratus       (202 images)
- Contrails     (200 images)
Total :         2543 images
---
## Predictions for each class:
- Cirrus        : A warm front is approaching.
- Cirrostratus  : A storm is coming.
- Cirrocumulus  : The weather is about to change!
- Altocumulus   : Rain is coming soon!
- Altostratus   : Rain, incoming.
- Cumulus       : Fair weather!
- Cumulonimbus  : Thunderstorms may be due.
- Nimbostratus  : Rain / Fog incoming.
- Stratocumulus : Bad weather incoming.
- Stratus       : Light rain.
- Contrails     : Airplanes!

Credits : https://www.boatus.com/magazine/2018/april/how-to-read-clouds.asp
---

To balance the dataset, we'll assume that we need 350 images in each class (since max + 10)

So, extending this, each class requires
- Cirrus        (211 images)
- Cirrostratus  (063 images)
- Cirrocumulus  (082 images)
- Altocumulus   (129 images)
- Altostratus   (162 images)
- Cumulus       (168 images)
- Cumulonimbus  (108 images)
- Nimbostratus  (076 images)
- Stratocumulus (010 images)
- Stratus       (148 images)
- Contrails     (150 images)
Total :         3850 images

Note:
Since all we're doing is looking for patterns and not features, we're assuming that augmenting a class to
more than double the images it initially started off with won't have a serious impact on the final results.
This may/may not be a valid assumption, though. We'll only know for sure after training.
Also, since all we're looking for is a pattern, rotation and flipping don't make much sense. Same goes for
adding noise and blurring. And given how some images have buildings in the bottom, translating is also risky.
Hence, we're only cropping or resizing the images.
