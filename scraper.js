const gpRaw = require('google-play-scraper');
const gplay = gpRaw.default || gpRaw;
const asRaw = require('app-store-scraper');
const appStore = asRaw.default || asRaw;
const fs = require('fs');
const path = require('path');

const config = require('./config.json');

async function scrapeReviews() {
    let allReviews = [];

    console.log(`Fetching Google Play reviews for ${config.playStoreAppId}...`);
    try {
        const playReviews = await gplay.reviews({
            appId: config.playStoreAppId,
            sort: gplay.sort.NEWEST,
            num: config.maxPlayStoreReviews
        });
        
        console.log(`Fetched ${playReviews.data.length} Google Play reviews.`);
        
        for (const r of playReviews.data) {
            allReviews.push({
                platform: 'Google Play',
                rating: r.score,
                date: new Date(r.date).toISOString(),
                text: r.text,
                thumbsUp: r.thumbsUp
            });
        }
    } catch (err) {
        console.error('Error fetching Google Play reviews:', err);
    }

    console.log(`Fetching App Store reviews for ${config.appStoreAppId}...`);
    for (const country of config.appStoreCountries) {
        for (let page = 1; page <= config.appStoreMaxPagesPerCountry; page++) {
            try {
                const asReviews = await appStore.reviews({
                    id: config.appStoreAppId,
                    sort: appStore.sort.RECENT,
                    page: page,
                    country: country
                });
                if (!asReviews || asReviews.length === 0) break;

                for (const r of asReviews) {
                    allReviews.push({
                        platform: 'App Store',
                        rating: Number(r.score),
                        date: new Date(r.updated || r.date || Date.now()).toISOString(),
                        text: r.text,
                        thumbsUp: 0
                    });
                }
            } catch (err) {
                break; // Stop paginating if error
            }
        }
    }
    
    // Deduplicate App Store reviews by text, since pulling from multiple countries might overlap
    const uniqueReviewsMap = new Map();
    for (const r of allReviews) {
        // Use text + platform as a weak deduplication key
        const key = r.platform + '|' + r.text;
        if (!uniqueReviewsMap.has(key)) {
            uniqueReviewsMap.set(key, r);
        }
    }
    allReviews = Array.from(uniqueReviewsMap.values());

    console.log(`Total unique reviews fetched: ${allReviews.length}`);
    
    const outputPath = path.join(__dirname, 'reviews.json');
    fs.writeFileSync(outputPath, JSON.stringify(allReviews, null, 2));
    console.log(`Saved reviews to ${outputPath}`);
}

scrapeReviews();
