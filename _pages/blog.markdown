---
layout: single
#title:  "Blog"
permalink: /blog/
author_profile: true
comments: false
description: My collection of projects and machine learning topics
---

I'm gathering some topics that I've found useful on Machine Learning. Hope that you find them interesting too.

<ul>
  {% for post in site.posts %}
    {% unless post.next %}
      <font color="#778899"><h2>{{ post.date | date: '%Y %b' }}</h2></font>
    {% else %}
      {% capture year %}{{ post.date | date: '%Y %b' }}{% endcapture %}
      {% capture nyear %}{{ post.next.date | date: '%Y %b' }}{% endcapture %}
      {% if year != nyear %}
        <font color="#778899"><h2>{{ post.date | date: '%Y %b' }}</h2></font>
      {% endif %}

    {% endunless %}
   {% include archive-single.html %}
  {% endfor %}
</ul>
