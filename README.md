# steel-stealth-model

One of, if not the most important part of Agents getting to browse the internet is to not be detected by the target's security systems. This means elliciting captchas, and staying overall stealthy by using techniques such as IP rotation, browser fingerprinting, and user agent spoofing. Additionally, Agents must also be able to navigate the web without leaving a trace, which means using tools such as VPNs and proxy servers to mask their location and identity.

At Steel, we collect a lot of data on our browsers stealth such as user agent spoofing, IP rotation, and browser fingerprinting. I wanted to use this data to create a model that would create a User Agent, fingerprint, stealth config to avoid captcha challenges.

**GOAL**: Minimize CAPTCHA challenges

Steps:
1. So first I will need to gather the data from Steel's session database on a couple different important factors. Currently we don't collect data on fingerprints so that will need to be added, websites would be important to for post-training, plus if a captcha exists on page which I'll need to figure out how to check if it exists. This would be very important as that would dictacte the training data.
  a. User Agent String
  b. Stealth Config
  c. Browser Fingerprint
  d. Captcha exists on page
  e. Captcha invoked - Reward function
  f. **Whatever else we want to include**

2. Captcha solved - Reward function
So to make this model make sense, I want to minimize the chance of capthca challenges appearing. This is possible and also solving captchas are possible as well, but to minimize and improve stealth is all the better.

3. Normalize the data
I will need to find a way to normalize the strings (User Agent, Browser Fingerprint, Stealth Config) as to make it something that a model can understand.

4. Teacher forcing pre-training the model
I will need to train the model on the normalized data to create a User Agent, fingerprint, stealth config to avoid captcha challenges. This is to get the model to learn the patterns and relationships between the data and generate good defaults.

5. Reinforcement Learning
I will feed the model some information on the website (URL, HTML content, metadata, cookies, headers) and then train it to generate User Agent, fingerprint, stealth config to avoid captcha challenges by minimizing the reward function.

6a. Post-train the model (Playground)
I will setup a playground where the model will generate User Agent, fingerprint, stealth config to avoid captcha challenges and then traverse to websites that usually or have required captchas in the past. This will allow us to have some continual learning as the more websites that we traverse, the more data that we can collect/train/test to optimize for more stealth/less captcha challenges.

6b. Telemetry
We can set up a telemetry system on steel-browser that then collects data on the OS version as well to increase our data surface area.

7. Deployment
We can deploy this model in Steel-browser as a give back to the community.

SUCCESS!!
