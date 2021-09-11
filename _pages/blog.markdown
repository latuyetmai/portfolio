---
layout: single
#title:  "Blog"
permalink: /blog/
author_profile: true
comments: false
description: My collection of projects and machine learning topics
---

Machine learning is a rapidly growing field at the intersection of computer science and statistics and concerned with finding patterns in data. It is responsible for tremendous advances in technology, from personalized product recommendations to speech recognition in smart devices. 

The emphasis of this blog will be on intuition and practical examples of different Machine Learning algorithms and techniques.

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
