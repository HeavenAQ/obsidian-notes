// Notion Property Tags — generated for this vault.
// Directly styles tag pills inside page properties, independent of the Colored Tags plugin selectors.
const { Plugin } = require('obsidian');

const COLOR_MAP = {
  "recommended": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "to-read": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "multimodal": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "domain-generalization": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "concept": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "to-review": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "self-contact-hand-face-cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "lesson-summary": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "not-started": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "done": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "cognition": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "dataset": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "theory": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "contact-aware": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "paradigms": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "hand-face-cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "in-progress": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "turn-taking": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "dataset-corpus": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "essential": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "multimodal-disfluency-and-discourse": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "background": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "dynamic-programming": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "gesture-and-turn-taking": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "benchmark": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "cvpr": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "leetcode": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "medium": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "problem": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "reading": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "to-solve": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "arxiv": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "disfluency": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "gpu": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "none": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "skim-skip": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "ml": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "not-relevant": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "action-recognition-backbone": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "hardware": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "llm": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "data-structures": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "journal": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "lesson-day": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "read": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "adaptor-and-self-touch-theory": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "pose-estimation": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "self-adaptor": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "weakly": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "iccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "sorting": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "gesture": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "graphs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "icmi": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "linux": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "strongly": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "video-backbone": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "aigc": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "egocentric": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "finetuning": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundational": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "icassp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "object-detection": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "survey": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "weak-supervision": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "aaai": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "architecture": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "challenges": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "distress": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "fundamentals": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "geometry": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "iclr": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "interspeech": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "neurips": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "self-touch": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "skeleton": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "wacv-iccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "advanced": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "advanced-lecture": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundations-of-learning": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "generative-and-representation": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "languages": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "linear-filters": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "lrec": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "motion": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "plos-one": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "repair": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "resnet": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "sampling-and-multiscale": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "searching": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "sft": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "shortest-path": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "acii": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "acl-hlt": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "acm-mm": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "acm-tiis": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "augmented-humans": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "backtracking": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "bfs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "binary-search": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "chi": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "cogn-affect-behav-neurosci": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "complexity": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "computer-speech-language": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "dfs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "discourse": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "diss": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "divide-and-conquer": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "eccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "emnlp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "findings-of-acl": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "findings-of-emnlp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundations": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "graph": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "greedy": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "hash-table": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "heap": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "ieee-acm-taslp": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-fg": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-affective-computing": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-ind-electron": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-instrum-meas": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-intell-transp-syst": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ijcnlp": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "image-formation": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "image-processing": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "iui": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "iwsds": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "japanese": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "jslhr": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "language": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "linked-list": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "method": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "mlmi": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "naacl": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "naacl-workshop": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "neural-architectures": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "neurocognitive": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "phil-trans-r-soc-b": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "prefix-sum": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "probabilistic-models": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "queue": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "recursion": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "research-and-closing": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "scientific-reports": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "segment-tree": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "self-collected": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "semiotica": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "siggraph-asia": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "sliding-window": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "soc-sci-and-medicine": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "stack": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "synthese": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "tree": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "trie": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "union-find": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "vision-and-language": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "channels": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "pixels": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "thedress": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  }
};
const DEFAULT = {
  light: { bg: '#F1F1EF', fg: '#37352F' },
  dark: { bg: '#2F2F2C', fg: '#D4D4D4' },
  name: 'default'
};

function normalizeTagText(text) {
  if (!text) return null;
  return String(text)
    .replace(/[×✕✖]\s*$/g, '')
    .replace(/^#/, '')
    .trim()
    .replace(/\s+/g, '-')
    .toLowerCase() || null;
}

function pillText(el) {
  const content = el.querySelector('.multi-select-pill-content, [class*="pill-content"], [class*="select-pill-content"]');
  if (content) return content.textContent;
  const clone = el.cloneNode(true);
  clone.querySelectorAll('.multi-select-pill-remove-button, [class*="remove"], button, svg').forEach((x) => x.remove());
  return clone.textContent;
}

function setImportant(el, prop, value) {
  el.style.setProperty(prop, value, 'important');
}

function stylePill(el, colors, tag) {
  const { bg, fg } = colors;
  el.classList.add('notion-property-tag');
  el.classList.add(`notion-property-tag-${tag}`);
  setImportant(el, '--tag-color', fg);
  setImportant(el, '--tag-color-hover', fg);
  setImportant(el, '--tag-background', bg);
  setImportant(el, '--tag-background-hover', bg);
  setImportant(el, 'color', fg);
  setImportant(el, '-webkit-text-fill-color', fg);
  setImportant(el, 'background', bg);
  setImportant(el, 'background-color', bg);
  setImportant(el, 'background-image', 'none');
  setImportant(el, 'border-color', 'transparent');
  setImportant(el, 'border-radius', '4px');
  setImportant(el, 'border-width', '0');
  setImportant(el, 'box-shadow', 'none');
  setImportant(el, 'font-weight', '500');
  setImportant(el, 'font-size', '0.88em');
  setImportant(el, 'line-height', '1.45');
  setImportant(el, 'padding', '1px 6px');
  el.querySelectorAll('*').forEach((child) => {
    setImportant(child, 'color', fg);
    setImportant(child, '-webkit-text-fill-color', fg);
    setImportant(child, 'fill', fg);
    setImportant(child, 'stroke', fg);
  });
}

function candidates(root = document.body) {
  const selectors = [
    '.metadata-property .multi-select-pill',
    '.metadata-container .multi-select-pill',
    '.metadata-properties .multi-select-pill',
    '.metadata-property [class*="multi-select-pill"]',
    '.metadata-property-value [class*="multi-select-pill"]',
    '.metadata-property [class*="select-pill"]',
    '.metadata-property-value [class*="select-pill"]'
  ].join(', ');
  const out = [];
  if (root instanceof HTMLElement && root.matches(selectors)) out.push(root);
  if (root.querySelectorAll) root.querySelectorAll(selectors).forEach((el) => out.push(el));
  return [...new Set(out)].filter((el) => !String(el.className).includes('remove-button'));
}

module.exports = class NotionPropertyTagsPlugin extends Plugin {
  onload() {
    this.apply = this.apply.bind(this);
    this.schedule = this.schedule.bind(this);
    this.apply();
    this.observer = new MutationObserver(this.schedule);
    this.observer.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['class'] });
    this.registerEvent(this.app.workspace.on('active-leaf-change', this.schedule));
    this.registerEvent(this.app.workspace.on('layout-change', this.schedule));
    this.registerEvent(this.app.workspace.on('file-open', this.schedule));
    this.registerDomEvent(window, 'focus', this.schedule);
  }

  onunload() {
    if (this.observer) this.observer.disconnect();
    if (this.raf) window.cancelAnimationFrame(this.raf);
  }

  schedule() {
    if (this.raf) return;
    this.raf = window.requestAnimationFrame(() => {
      this.raf = null;
      this.apply();
    });
  }

  apply(root = document.body) {
    const mode = document.body.classList.contains('theme-dark') ? 'dark' : 'light';
    for (const el of candidates(root)) {
      const tag = normalizeTagText(pillText(el));
      if (!tag) continue;
      const pair = COLOR_MAP[tag] || DEFAULT;
      stylePill(el, pair[mode], tag);
    }
  }
};
