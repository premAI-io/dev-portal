---
id: install-aws-ec2.md
title: Install PremAI on AWS EC2
sidebar_label: Install on AWS EC2
sidebar_position: 3
---

Launch an EC2 instance using the below bookmark link, opening port 8000 and selecting the `g5.xlarge` instance type.


### Prerequisites:

1. An [active AWS account](https://docs.aws.amazon.com/SetUp/latest/UserGuide/setup-AWSsignup.html).
2. An [IAM user](https://docs.aws.amazon.com/SetUp/latest/UserGuide/setup-configadminuser.html) with [permissions to launch EC2 instances](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_ec2_region.html).

### Step by Step Guide:

```
https://console.aws.amazon.com/ec2/v2/home?region=eu-west-1#LaunchInstanceWizard:ami=ami-0c08919f39b2887c0
```

1. **Access the Bookmark Link**:
    - Open your browser.
    - Click on the [bookmark](https://console.aws.amazon.com/ec2/v2/home?region=eu-west-1#LaunchInstanceWizard:ami=ami-0c08919f39b2887c0) or paste the bookmark link into your browser's address bar and press Enter. 
    - This will take you to the AWS Management Console, specifically to the EC2 Launch Instance Wizard with the public AMI pre-selected.

2. **Choose an Instance Type**:
    - Scroll down the list until you find `g5.xlarge`. 
    - Click on the `Select` button next to it.

3. **Configure Instance Details**:
    - Usually, the default settings in this step are okay for most use-cases, but you can make changes if needed.

4. **Add Storage**:
    - Add any storage volumes if necessary or keep the default. Suggested `64GB` to try enough Prem services.

5. **Add Tags**:
    - You can add key-value pairs to help identify your instance later on, but this is optional.

6. **Configure Security Group**:
    - Choose “Create a new security group”.
    - You will see that by default SSH port 22 is open.
    - Click on the `Add Rule` button to add more rules.
        - For HTTP: Choose HTTP from the dropdown and it will automatically populate the port range as `80`.
        - For HTTPS: Choose HTTPS from the dropdown and it will automatically populate the port range as `443`.
        - For port 8000: Choose `Custom TCP Rule` from the dropdown, type `8000` in the port range box, and choose 'Anywhere' (or a specific IP or range) in the source box.
    
    Remember: Be cautious when opening ports to 'Anywhere' as it exposes those ports to the entire internet. Consider tightening security by only allowing specific IPs or IP ranges.

7. **Review and Launch**:
    - Review your settings to ensure everything is correct.
    - Click on the `Launch` button.
    
8. **Key Pair**:
    - A prompt will appear asking you to select a key pair or create a new one.
        - If you already have a key pair and want to use it: Select "Choose an existing key pair", then select the desired key pair from the dropdown list.
        - If you need a new key pair: Select "Create a new key pair", name it, and download the key pair. Make sure to save it in a safe place, as AWS won’t give you the private key again.
    - After selecting or creating the key pair, check the acknowledgment box that says you have access to the selected private key file.
    - Finally, click the `Launch Instances` button.

9. **Confirmation Page**:
    - After launching, you'll be taken to a confirmation page where you can click on the `View Instances` button to see your new instance starting up.

That's it! After a few minutes, your EC2 instance should be up and running.