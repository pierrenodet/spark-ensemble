"use strict";(self.webpackChunkspark_ensemble=self.webpackChunkspark_ensemble||[]).push([[674],{3905:function(e,a,r){r.d(a,{Zo:function(){return p},kt:function(){return m}});var n=r(7294);function t(e,a,r){return a in e?Object.defineProperty(e,a,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[a]=r,e}function i(e,a){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),r.push.apply(r,n)}return r}function l(e){for(var a=1;a<arguments.length;a++){var r=null!=arguments[a]?arguments[a]:{};a%2?i(Object(r),!0).forEach((function(a){t(e,a,r[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(r,a))}))}return e}function s(e,a){if(null==e)return{};var r,n,t=function(e,a){if(null==e)return{};var r,n,t={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],a.indexOf(r)>=0||(t[r]=e[r]);return t}(e,a);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],a.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(t[r]=e[r])}return t}var o=n.createContext({}),c=function(e){var a=n.useContext(o),r=a;return e&&(r="function"==typeof e?e(a):l(l({},a),e)),r},p=function(e){var a=c(e.components);return n.createElement(o.Provider,{value:a},e.children)},d={inlineCode:"code",wrapper:function(e){var a=e.children;return n.createElement(n.Fragment,{},a)}},u=n.forwardRef((function(e,a){var r=e.components,t=e.mdxType,i=e.originalType,o=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=c(r),m=t,g=u["".concat(o,".").concat(m)]||u[m]||d[m]||i;return r?n.createElement(g,l(l({ref:a},p),{},{components:r})):n.createElement(g,l({ref:a},p))}));function m(e,a){var r=arguments,t=a&&a.mdxType;if("string"==typeof e||t){var i=r.length,l=new Array(i);l[0]=u;var s={};for(var o in a)hasOwnProperty.call(a,o)&&(s[o]=a[o]);s.originalType=e,s.mdxType="string"==typeof e?e:t,l[1]=s;for(var c=2;c<i;c++)l[c]=r[c];return n.createElement.apply(null,l)}return n.createElement.apply(null,r)}u.displayName="MDXCreateElement"},5177:function(e,a,r){r.r(a),r.d(a,{assets:function(){return p},contentTitle:function(){return o},default:function(){return m},frontMatter:function(){return s},metadata:function(){return c},toc:function(){return d}});var n=r(7462),t=r(3366),i=(r(7294),r(3905)),l=["components"],s={id:"exemple",title:"Exemple"},o=void 0,c={unversionedId:"exemple",id:"exemple",title:"Exemple",description:"Building Base Learner",source:"@site/../spark-ensemble-docs/target/mdoc/exemple.md",sourceDirName:".",slug:"/exemple",permalink:"/spark-ensemble/docs/exemple",draft:!1,editUrl:"https://github.com/pierrenodet/spark-ensemble/edit/master/docs/exemple.md",tags:[],version:"current",frontMatter:{id:"exemple",title:"Exemple"},sidebar:"someSidebar",previous:{title:"GBM",permalink:"/spark-ensemble/docs/gbm"}},p={},d=[{value:"Building Base Learner",id:"building-base-learner",level:2},{value:"Building Meta Estimator",id:"building-meta-estimator",level:2},{value:"Building Param Grid",id:"building-param-grid",level:2},{value:"Grid Search with Cross Validation",id:"grid-search-with-cross-validation",level:2},{value:"Save and Load",id:"save-and-load",level:2}],u={toc:d};function m(e){var a=e.components,r=(0,t.Z)(e,l);return(0,i.kt)("wrapper",(0,n.Z)({},u,r,{components:a,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"building-base-learner"},"Building Base Learner"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-scala"},"import org.apache.spark.ml.classification.DecisionTreeClassifier\n\nval baseClassifier = new DecisionTreeClassifier()\n.setMaxDepth(20)\n")),(0,i.kt)("h2",{id:"building-meta-estimator"},"Building Meta Estimator"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-scala"},"import org.apache.spark.ml.classification.BaggingClassifier\n\nval baggingClassifier = new BaggingClassifier()\n.setBaseLearner(baseClassifier)\n.setNumBaseLearners(10)\n.setParallelism(4)\n")),(0,i.kt)("h2",{id:"building-param-grid"},"Building Param Grid"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-scala"},"import org.apache.spark.ml.tuning.ParamGridBuilder\n\nval paramGrid = new ParamGridBuilder()\n        .addGrid(baggingClassifier.numBaseLearners, Array(10,20))\n        .addGrid(baseClassifier.maxDepth, Array(10,20))\n        .build()\n")),(0,i.kt)("h2",{id:"grid-search-with-cross-validation"},"Grid Search with Cross Validation"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-scala"},"import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator\nimport org.apache.spark.ml.tuning.CrossValidator\nimport org.apache.spark.ml.classification.BaggingClassificationModel\n\nval cv = new CrossValidator()\n        .setEstimator(baggingClassifier)\n        .setEvaluator(new MulticlassClassificationEvaluator())\n        .setEstimatorParamMaps(paramGrid)\n        .setNumFolds(5)\n        .setParallelism(4)\n\nval cvModel = cv.fit(data)\n\nval bestModel = cvModel.bestModel.asInstanceOf[BaggingClassificationModel]\n\nbestModel\n")),(0,i.kt)("h2",{id:"save-and-load"},"Save and Load"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-scala"},'bestModel.write.overwrite().save("/tmp/model")\nval loaded = BaggingClassificationModel.load("/tmp/model")\n')))}m.isMDXComponent=!0}}]);