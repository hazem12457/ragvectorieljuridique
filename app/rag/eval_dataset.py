"""Jeu d'évaluation (questions en français, vérité terrain = identifiants d'articles du corpus).

Chaque item :
    question  : question en langage naturel (sans donner le numéro d'article la plupart du temps)
    type      : factuelle | relationnelle | multi_hop
    gold      : articles qui contiennent la réponse  (-> Recall@k, MRR)
    reference : réponse attendue, rédigée à partir du texte officiel (-> RAGAS context_recall)

Toutes les références ont été vérifiées dans data/eurlex/articles.jsonl.
Jeu de départ : 24 questions. À compléter à 30–40 lors de l'étape « Évaluation »
(en ajoutant surtout des questions relationnelles et multi-hop, où le Graph RAG
devrait faire la différence).
"""

EVAL_QUESTIONS = [
    # ------------------------------------------------------------------ factuelles
    {
        "id": "F01", "type": "factuelle",
        "question": "Dans quel délai le responsable du traitement doit-il notifier une violation de données personnelles à l'autorité de contrôle selon le RGPD ?",
        "gold": ["32016R0679_art_33"],
        "reference": "Dans les meilleurs délais et, si possible, 72 heures au plus tard après en avoir pris connaissance, sauf si la violation n'est pas susceptible d'engendrer un risque pour les droits et libertés (article 33 du RGPD).",
    },
    {
        "id": "F02", "type": "factuelle",
        "question": "Quel est le montant maximal des amendes administratives prévues par le RGPD ?",
        "gold": ["32016R0679_art_83"],
        "reference": "Jusqu'à 20 000 000 EUR ou, pour une entreprise, jusqu'à 4 % du chiffre d'affaires annuel mondial total de l'exercice précédent, le montant le plus élevé étant retenu ; un palier inférieur de 10 000 000 EUR ou 2 % s'applique à certaines violations (article 83 du RGPD).",
    },
    {
        "id": "F03", "type": "factuelle",
        "question": "Dans quels cas une personne peut-elle obtenir l'effacement de ses données personnelles ?",
        "gold": ["32016R0679_art_17"],
        "reference": "Article 17 du RGPD (droit à l'oubli) : notamment lorsque les données ne sont plus nécessaires, que la personne retire son consentement, s'oppose au traitement, que les données ont fait l'objet d'un traitement illicite ou doivent être effacées pour respecter une obligation légale.",
    },
    {
        "id": "F04", "type": "factuelle",
        "question": "Quelles sont les bases légales qui rendent licite un traitement de données à caractère personnel ?",
        "gold": ["32016R0679_art_6"],
        "reference": "Article 6 du RGPD : consentement, exécution d'un contrat, obligation légale, sauvegarde des intérêts vitaux, mission d'intérêt public ou exercice de l'autorité publique, intérêts légitimes du responsable du traitement ou d'un tiers.",
    },
    {
        "id": "F05", "type": "factuelle",
        "question": "Quand un organisme est-il obligé de désigner un délégué à la protection des données ?",
        "gold": ["32016R0679_art_37"],
        "reference": "Article 37 du RGPD : lorsque le traitement est effectué par une autorité ou un organisme public (hors juridictions), lorsque les activités de base consistent en un suivi régulier et systématique à grande échelle des personnes, ou en un traitement à grande échelle de catégories particulières de données ou de données relatives aux condamnations pénales.",
    },
    {
        "id": "F06", "type": "factuelle",
        "question": "Sous quel format les données doivent-elles être fournies au titre du droit à la portabilité ?",
        "gold": ["32016R0679_art_20"],
        "reference": "Dans un format structuré, couramment utilisé et lisible par machine, avec le droit de les transmettre à un autre responsable du traitement (article 20 du RGPD).",
    },
    {
        "id": "F07", "type": "factuelle",
        "question": "Quelles pratiques en matière d'intelligence artificielle sont interdites par l'AI Act ?",
        "gold": ["32024R1689_art_5"],
        "reference": "Article 5 de l'AI Act : notamment les techniques subliminales ou manipulatrices, l'exploitation des vulnérabilités, la notation sociale, certaines évaluations du risque d'infraction, la constitution de bases de reconnaissance faciale par moissonnage non ciblé, la reconnaissance des émotions au travail et à l'école, certaines catégorisations biométriques et l'identification biométrique à distance en temps réel à des fins répressives (sauf exceptions).",
    },
    {
        "id": "F08", "type": "factuelle",
        "question": "Quelles amendes sont encourues en cas de non-respect des pratiques d'IA interdites ?",
        "gold": ["32024R1689_art_99"],
        "reference": "Jusqu'à 35 000 000 EUR ou, pour une entreprise, jusqu'à 7 % du chiffre d'affaires annuel mondial total de l'exercice précédent, le montant le plus élevé étant retenu (article 99 de l'AI Act).",
    },
    {
        "id": "F09", "type": "factuelle",
        "question": "À partir de quelle date l'AI Act est-il applicable, et quelles dispositions s'appliquent plus tôt ?",
        "gold": ["32024R1689_art_113"],
        "reference": "Il est applicable à partir du 2 août 2026 ; les chapitres I et II s'appliquent dès le 2 février 2025, et certaines dispositions (chapitre III section 4, chapitres V, VII, XII, article 78) dès le 2 août 2025 (article 113).",
    },
    {
        "id": "F10", "type": "factuelle",
        "question": "À partir de combien d'utilisateurs une plateforme est-elle considérée comme une très grande plateforme en ligne au sens du DSA ?",
        "gold": ["32022R2065_art_33"],
        "reference": "Lorsqu'elle a un nombre mensuel moyen de destinataires actifs dans l'Union égal ou supérieur à 45 millions et qu'elle est désignée par la Commission (article 33 du DSA).",
    },
    {
        "id": "F11", "type": "factuelle",
        "question": "Quels seuils font présumer qu'une entreprise est un contrôleur d'accès selon le DMA ?",
        "gold": ["32022R1925_art_3"],
        "reference": "Notamment un chiffre d'affaires annuel dans l'Union d'au moins 7,5 milliards d'euros sur chacun des trois derniers exercices (ou une capitalisation boursière d'au moins 75 milliards), et un service de plateforme essentiel comptant au moins 45 millions d'utilisateurs finaux actifs par mois et 10 000 entreprises utilisatrices actives par an dans l'Union (article 3 du DMA).",
    },
    {
        "id": "F12", "type": "factuelle",
        "question": "Quelles sont les étapes et délais de notification d'un incident important prévus par NIS2 ?",
        "gold": ["32022L2555_art_23"],
        "reference": "Une alerte précoce dans les 24 heures, une notification d'incident dans les 72 heures, un rapport intermédiaire sur demande et un rapport final au plus tard un mois après la notification (article 23 de NIS2).",
    },
    {
        "id": "F13", "type": "factuelle",
        "question": "Dans quel délai un fabricant doit-il signaler une vulnérabilité activement exploitée dans un produit comportant des éléments numériques ?",
        "gold": ["32024R2847_art_14"],
        "reference": "Une alerte précoce au plus tard 24 heures après en avoir eu connaissance, puis une notification au plus tard 72 heures après (article 14 du Cyber Resilience Act).",
    },
    {
        "id": "F14", "type": "factuelle",
        "question": "Quelles obligations de transparence s'appliquent aux systèmes d'IA qui génèrent des hypertrucages (deepfakes) ?",
        "gold": ["32024R1689_art_50"],
        "reference": "Les déployeurs d'un système d'IA qui génère ou manipule des images, contenus audio ou vidéo constituant un hypertrucage indiquent que les contenus ont été générés ou manipulés par une IA ; les fournisseurs marquent les contenus de synthèse dans un format lisible par machine (article 50 de l'AI Act).",
    },
    # ------------------------------------------------------------------ relationnelles
    {
        "id": "R01", "type": "relationnelle",
        "question": "Quelle directive le RGPD abroge-t-il et à partir de quand ?",
        "gold": ["32016R0679_art_94"],
        "reference": "La directive 95/46/CE, abrogée avec effet au 25 mai 2018 ; les références à la directive s'entendent comme faites au RGPD (article 94).",
    },
    {
        "id": "R02", "type": "relationnelle",
        "question": "Comment le DSA modifie-t-il la directive sur le commerce électronique 2000/31/CE ?",
        "gold": ["32022R2065_art_89"],
        "reference": "Les articles 12 à 15 de la directive 2000/31/CE sont supprimés et les références à ces articles s'entendent comme faites aux articles 4, 5, 6 et 8 du DSA (article 89).",
    },
    {
        "id": "R03", "type": "relationnelle",
        "question": "Quelle directive est abrogée par NIS2 ?",
        "gold": ["32022L2555_art_44"],
        "reference": "La directive (UE) 2016/1148 (NIS 1), abrogée avec effet au 18 octobre 2024 (article 44 de NIS2).",
    },
    {
        "id": "R04", "type": "relationnelle",
        "question": "Sur quels articles du RGPD s'appuie l'obligation de réaliser une analyse d'impact, et quand est-elle requise ?",
        "gold": ["32016R0679_art_35"],
        "reference": "Article 35 du RGPD : une analyse d'impact est requise lorsqu'un traitement est susceptible d'engendrer un risque élevé, notamment en cas d'évaluation systématique fondée sur un traitement automatisé, de traitement à grande échelle de données sensibles (article 9) ou de surveillance systématique à grande échelle d'une zone accessible au public.",
    },
    # ------------------------------------------------------------------ multi-hop
    {
        "id": "M01", "type": "multi_hop",
        "question": "Comparez les délais de notification d'incident du RGPD et de NIS2.",
        "gold": ["32016R0679_art_33", "32022L2555_art_23"],
        "reference": "RGPD : notification de la violation à l'autorité de contrôle si possible dans les 72 heures (article 33). NIS2 : alerte précoce sous 24 heures, notification sous 72 heures et rapport final sous un mois (article 23).",
    },
    {
        "id": "M02", "type": "multi_hop",
        "question": "Quels sont les plafonds d'amende du RGPD, de l'AI Act et du DMA ?",
        "gold": ["32016R0679_art_83", "32024R1689_art_99", "32022R1925_art_30"],
        "reference": "RGPD : jusqu'à 20 M EUR ou 4 % du CA mondial (article 83). AI Act : jusqu'à 35 M EUR ou 7 % pour les pratiques interdites (article 99). DMA : jusqu'à 10 % du CA mondial, 20 % en cas de récidive (article 30).",
    },
    {
        "id": "M03", "type": "multi_hop",
        "question": "Quelles obligations pèsent sur les très grandes plateformes en ligne : à partir de quand sont-elles désignées et quels risques doivent-elles évaluer ?",
        "gold": ["32022R2065_art_33", "32022R2065_art_34"],
        "reference": "Désignation au-delà de 45 millions de destinataires actifs mensuels (article 33 du DSA) ; elles recensent, analysent et évaluent les risques systémiques : diffusion de contenus illicites, effets sur les droits fondamentaux, le discours civique et les processus électoraux, la sécurité publique, la santé et les mineurs (article 34).",
    },
    {
        "id": "M04", "type": "multi_hop",
        "question": "Comparez les obligations de signalement d'incidents de NIS2 et du Cyber Resilience Act.",
        "gold": ["32022L2555_art_23", "32024R2847_art_14"],
        "reference": "NIS2 (article 23) : entités essentielles et importantes, alerte précoce 24 h, notification 72 h, rapport final un mois. CRA (article 14) : fabricants, vulnérabilités activement exploitées et incidents graves, alerte précoce 24 h et notification 72 h.",
    },
    {
        "id": "M05", "type": "multi_hop",
        "question": "Quelles amendes les États membres doivent-ils prévoir pour les entités essentielles et importantes sous NIS2, et comment se comparent-elles aux amendes du RGPD ?",
        "gold": ["32022L2555_art_34", "32016R0679_art_83"],
        "reference": "NIS2 (article 34) : maximum d'au moins 10 M EUR ou 2 % du CA mondial pour les entités essentielles, au moins 7 M EUR ou 1,4 % pour les entités importantes. RGPD (article 83) : jusqu'à 20 M EUR ou 4 % (10 M EUR ou 2 % pour le palier inférieur).",
    },
    {
        "id": "M06", "type": "multi_hop",
        "question": "Quel droit d'accès aux données des produits connectés le Data Act donne-t-il à l'utilisateur, et peut-il les partager avec un tiers ?",
        "gold": ["32023R2854_art_3", "32023R2854_art_4", "32023R2854_art_5"],
        "reference": "Les produits connectés sont conçus pour rendre les données accessibles à l'utilisateur (article 3) ; à défaut d'accès direct, le détenteur met les données à disposition de l'utilisateur sur demande (article 4) ; à la demande de l'utilisateur, le détenteur les met à disposition d'un tiers (article 5).",
    },
]
