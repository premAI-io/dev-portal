import React from 'react';
import clsx from 'clsx';
import {blogPostContainerID} from '@docusaurus/utils-common';
import {useBlogPost} from '@docusaurus/theme-common/internal';
import MDXContent from '@theme/MDXContent';
import {useBaseUrlUtils} from "@docusaurus/useBaseUrl";

export default function BlogPostItemContent({children, className}) {
  const {isBlogPostPage, frontMatter, assets} = useBlogPost();
  const {withBaseUrl} = useBaseUrlUtils();
  const image = assets.image ?? frontMatter.image;

  return (
    <div
      // This ID is used for the feed generation to locate the main content
      id={isBlogPostPage ? blogPostContainerID : undefined}
      className={clsx('markdown', className)}
      itemProp="articleBody">
      {isBlogPostPage ?
        <MDXContent>{children}</MDXContent>
        : (
          <div style={{display: "flex", gap: '2%'}}>
            <p style={{flexBasis: '66.66%'}}>{frontMatter.description}</p>
            <p style={{flexBasis: '33.33%'}}>
              <img src={withBaseUrl(image, {absolute: true})} alt=""/>
            </p>
          </div>
        )}
    </div>
  );
}
